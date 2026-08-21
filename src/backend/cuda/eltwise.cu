/// @file eltwise.cu
/// @brief CUDA implementation of element-wise binary operations.
///
/// Element-wise operations — flat grid-stride loop for rank < 2,
/// row-by-row pitch-aware kernel for rank >= 2 (supports packed layouts).
/// All 4 operations: Add, Sub, Mul, Div.
///
/// Supports f32 and f16. Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Row-by-row CUDA kernel — pitch-aware (rank >= 2)
// ============================================================

template <typename T>
__global__ void eltwise_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t num_rows,
    int64_t last_dim,
    int64_t a_row_stride,
    int64_t b_row_stride,
    int64_t o_row_stride,
    EltwiseType op_type,
    bool add_to)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4;  // 128-bit / sizeof(T)

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows) return;

    const T* a_row = a + row * a_row_stride;
    const T* b_row = b + row * b_row_stride;
    T* o_row = output + row * o_row_stride;

    const int64_t vec_end = (last_dim / VEC) * VEC;

    switch (op_type) {
    // ================================================================
    // Add
    // ================================================================
    case EltwiseType::Add: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                float4 va = v4_load(&a_row[i]);
                float4 vb = v4_load(&b_row[i]);
                if (add_to) {
                    float4 vo = v4_load(&o_row[i]);
                    v4_store(&o_row[i], make_float4(
                        vo.x + va.x + vb.x, vo.y + va.y + vb.y,
                        vo.z + va.z + vb.z, vo.w + va.w + vb.w));
                } else {
                    v4_store(&o_row[i], make_float4(
                        va.x + vb.x, va.y + vb.y,
                        va.z + vb.z, va.w + vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                half8 va = v4_load(&a_row[i]);
                half8 vb = v4_load(&b_row[i]);
                float r0 = __half2float(va.data[0]) + __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) + __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) + __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) + __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) + __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) + __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) + __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) + __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i], half8_add(v4_load(&o_row[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) + s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Sub
    // ================================================================
    case EltwiseType::Sub: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                float4 va = v4_load(&a_row[i]);
                float4 vb = v4_load(&b_row[i]);
                if (add_to) {
                    float4 vo = v4_load(&o_row[i]);
                    v4_store(&o_row[i], make_float4(
                        vo.x + va.x - vb.x, vo.y + va.y - vb.y,
                        vo.z + va.z - vb.z, vo.w + va.w - vb.w));
                } else {
                    v4_store(&o_row[i], make_float4(
                        va.x - vb.x, va.y - vb.y,
                        va.z - vb.z, va.w - vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                half8 va = v4_load(&a_row[i]);
                half8 vb = v4_load(&b_row[i]);
                float r0 = __half2float(va.data[0]) - __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) - __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) - __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) - __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) - __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) - __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) - __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) - __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i], half8_add(v4_load(&o_row[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) - s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Mul
    // ================================================================
    case EltwiseType::Mul: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                float4 va = v4_load(&a_row[i]);
                float4 vb = v4_load(&b_row[i]);
                if (add_to) {
                    float4 vo = v4_load(&o_row[i]);
                    v4_store(&o_row[i], make_float4(
                        vo.x + va.x * vb.x, vo.y + va.y * vb.y,
                        vo.z + va.z * vb.z, vo.w + va.w * vb.w));
                } else {
                    v4_store(&o_row[i], make_float4(
                        va.x * vb.x, va.y * vb.y,
                        va.z * vb.z, va.w * vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                half8 va = v4_load(&a_row[i]);
                half8 vb = v4_load(&b_row[i]);
                float r0 = __half2float(va.data[0]) * __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) * __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) * __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) * __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) * __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) * __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) * __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) * __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i], half8_add(v4_load(&o_row[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) * s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Div
    // ================================================================
    case EltwiseType::Div: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                float4 va = v4_load(&a_row[i]);
                float4 vb = v4_load(&b_row[i]);
                if (add_to) {
                    float4 vo = v4_load(&o_row[i]);
                    v4_store(&o_row[i], make_float4(
                        vo.x + va.x / vb.x, vo.y + va.y / vb.y,
                        vo.z + va.z / vb.z, vo.w + va.w / vb.w));
                } else {
                    v4_store(&o_row[i], make_float4(
                        va.x / vb.x, va.y / vb.y,
                        va.z / vb.z, va.w / vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                half8 va = v4_load(&a_row[i]);
                half8 vb = v4_load(&b_row[i]);
                float r0 = __half2float(va.data[0]) / __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) / __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) / __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) / __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) / __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) / __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) / __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) / __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i], half8_add(v4_load(&o_row[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
            float rv = s_load(&a_row[i]) / s_load(&b_row[i]);
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Min
    // ================================================================
    case EltwiseType::Min: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                float4 va = v4_load(&a_row[i]);
                float4 vb = v4_load(&b_row[i]);
                float r0 = fminf(va.x, vb.x);
                float r1 = fminf(va.y, vb.y);
                float r2 = fminf(va.z, vb.z);
                float r3 = fminf(va.w, vb.w);
                if (add_to) {
                    float4 vo = v4_load(&o_row[i]);
                    v4_store(&o_row[i], make_float4(
                        vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&o_row[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                half8 va = v4_load(&a_row[i]);
                half8 vb = v4_load(&b_row[i]);
                float r0 = fminf(__half2float(va.data[0]), __half2float(vb.data[0]));
                float r1 = fminf(__half2float(va.data[1]), __half2float(vb.data[1]));
                float r2 = fminf(__half2float(va.data[2]), __half2float(vb.data[2]));
                float r3 = fminf(__half2float(va.data[3]), __half2float(vb.data[3]));
                float r4 = fminf(__half2float(va.data[4]), __half2float(vb.data[4]));
                float r5 = fminf(__half2float(va.data[5]), __half2float(vb.data[5]));
                float r6 = fminf(__half2float(va.data[6]), __half2float(vb.data[6]));
                float r7 = fminf(__half2float(va.data[7]), __half2float(vb.data[7]));
                if (add_to) {
                    v4_store(&o_row[i], half8_add(v4_load(&o_row[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
            float av = s_load(&a_row[i]);
            float bv = s_load(&b_row[i]);
            float rv = av < bv ? av : bv;
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Max
    // ================================================================
    case EltwiseType::Max: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                float4 va = v4_load(&a_row[i]);
                float4 vb = v4_load(&b_row[i]);
                float r0 = fmaxf(va.x, vb.x);
                float r1 = fmaxf(va.y, vb.y);
                float r2 = fmaxf(va.z, vb.z);
                float r3 = fmaxf(va.w, vb.w);
                if (add_to) {
                    float4 vo = v4_load(&o_row[i]);
                    v4_store(&o_row[i], make_float4(
                        vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&o_row[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                half8 va = v4_load(&a_row[i]);
                half8 vb = v4_load(&b_row[i]);
                float r0 = fmaxf(__half2float(va.data[0]), __half2float(vb.data[0]));
                float r1 = fmaxf(__half2float(va.data[1]), __half2float(vb.data[1]));
                float r2 = fmaxf(__half2float(va.data[2]), __half2float(vb.data[2]));
                float r3 = fmaxf(__half2float(va.data[3]), __half2float(vb.data[3]));
                float r4 = fmaxf(__half2float(va.data[4]), __half2float(vb.data[4]));
                float r5 = fmaxf(__half2float(va.data[5]), __half2float(vb.data[5]));
                float r6 = fmaxf(__half2float(va.data[6]), __half2float(vb.data[6]));
                float r7 = fmaxf(__half2float(va.data[7]), __half2float(vb.data[7]));
                if (add_to) {
                    v4_store(&o_row[i], half8_add(v4_load(&o_row[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
            float av = s_load(&a_row[i]);
            float bv = s_load(&b_row[i]);
            float rv = av > bv ? av : bv;
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Pow
    // ================================================================
    case EltwiseType::Pow: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                float4 va = v4_load(&a_row[i]);
                float4 vb = v4_load(&b_row[i]);
                float r0 = powf(va.x, vb.x);
                float r1 = powf(va.y, vb.y);
                float r2 = powf(va.z, vb.z);
                float r3 = powf(va.w, vb.w);
                if (add_to) {
                    float4 vo = v4_load(&o_row[i]);
                    v4_store(&o_row[i], make_float4(
                        vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&o_row[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                half8 va = v4_load(&a_row[i]);
                half8 vb = v4_load(&b_row[i]);
                float r0 = powf(__half2float(va.data[0]), __half2float(vb.data[0]));
                float r1 = powf(__half2float(va.data[1]), __half2float(vb.data[1]));
                float r2 = powf(__half2float(va.data[2]), __half2float(vb.data[2]));
                float r3 = powf(__half2float(va.data[3]), __half2float(vb.data[3]));
                float r4 = powf(__half2float(va.data[4]), __half2float(vb.data[4]));
                float r5 = powf(__half2float(va.data[5]), __half2float(vb.data[5]));
                float r6 = powf(__half2float(va.data[6]), __half2float(vb.data[6]));
                float r7 = powf(__half2float(va.data[7]), __half2float(vb.data[7]));
                if (add_to) {
                    v4_store(&o_row[i], half8_add(v4_load(&o_row[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < last_dim; i += blockDim.x) {
            float rv = powf(s_load(&a_row[i]), s_load(&b_row[i]));
            if (add_to) {
                s_store(&o_row[i], s_load(&o_row[i]) + rv);
            } else {
                s_store(&o_row[i], rv);
            }
        }
        break;
    }
    }
}

// ============================================================
// Flat kernel for 1D / scalar tensors
// ============================================================

template <typename T>
__global__ void eltwise_flat_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ output,
    int64_t total,
    EltwiseType op_type,
    bool add_to)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4;  // 128-bit / sizeof(T)

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int64_t vec_end = (total / VEC) * VEC;

    switch (op_type) {
    // ================================================================
    // Add
    // ================================================================
    case EltwiseType::Add: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                float4 va = v4_load(&a[i]);
                float4 vb = v4_load(&b[i]);
                if (add_to) {
                    float4 vo = v4_load(&output[i]);
                    v4_store(&output[i], make_float4(
                        vo.x + va.x + vb.x, vo.y + va.y + vb.y,
                        vo.z + va.z + vb.z, vo.w + va.w + vb.w));
                } else {
                    v4_store(&output[i], make_float4(
                        va.x + vb.x, va.y + vb.y,
                        va.z + vb.z, va.w + vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                half8 va = v4_load(&a[i]);
                half8 vb = v4_load(&b[i]);
                float r0 = __half2float(va.data[0]) + __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) + __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) + __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) + __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) + __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) + __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) + __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) + __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i], half8_add(v4_load(&output[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < total; i += stride) {
            float rv = s_load(&a[i]) + s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Sub
    // ================================================================
    case EltwiseType::Sub: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                float4 va = v4_load(&a[i]);
                float4 vb = v4_load(&b[i]);
                if (add_to) {
                    float4 vo = v4_load(&output[i]);
                    v4_store(&output[i], make_float4(
                        vo.x + va.x - vb.x, vo.y + va.y - vb.y,
                        vo.z + va.z - vb.z, vo.w + va.w - vb.w));
                } else {
                    v4_store(&output[i], make_float4(
                        va.x - vb.x, va.y - vb.y,
                        va.z - vb.z, va.w - vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                half8 va = v4_load(&a[i]);
                half8 vb = v4_load(&b[i]);
                float r0 = __half2float(va.data[0]) - __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) - __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) - __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) - __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) - __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) - __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) - __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) - __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i], half8_add(v4_load(&output[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < total; i += stride) {
            float rv = s_load(&a[i]) - s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Mul
    // ================================================================
    case EltwiseType::Mul: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                float4 va = v4_load(&a[i]);
                float4 vb = v4_load(&b[i]);
                if (add_to) {
                    float4 vo = v4_load(&output[i]);
                    v4_store(&output[i], make_float4(
                        vo.x + va.x * vb.x, vo.y + va.y * vb.y,
                        vo.z + va.z * vb.z, vo.w + va.w * vb.w));
                } else {
                    v4_store(&output[i], make_float4(
                        va.x * vb.x, va.y * vb.y,
                        va.z * vb.z, va.w * vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                half8 va = v4_load(&a[i]);
                half8 vb = v4_load(&b[i]);
                float r0 = __half2float(va.data[0]) * __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) * __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) * __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) * __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) * __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) * __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) * __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) * __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i], half8_add(v4_load(&output[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < total; i += stride) {
            float rv = s_load(&a[i]) * s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Div
    // ================================================================
    case EltwiseType::Div: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                float4 va = v4_load(&a[i]);
                float4 vb = v4_load(&b[i]);
                if (add_to) {
                    float4 vo = v4_load(&output[i]);
                    v4_store(&output[i], make_float4(
                        vo.x + va.x / vb.x, vo.y + va.y / vb.y,
                        vo.z + va.z / vb.z, vo.w + va.w / vb.w));
                } else {
                    v4_store(&output[i], make_float4(
                        va.x / vb.x, va.y / vb.y,
                        va.z / vb.z, va.w / vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                half8 va = v4_load(&a[i]);
                half8 vb = v4_load(&b[i]);
                float r0 = __half2float(va.data[0]) / __half2float(vb.data[0]);
                float r1 = __half2float(va.data[1]) / __half2float(vb.data[1]);
                float r2 = __half2float(va.data[2]) / __half2float(vb.data[2]);
                float r3 = __half2float(va.data[3]) / __half2float(vb.data[3]);
                float r4 = __half2float(va.data[4]) / __half2float(vb.data[4]);
                float r5 = __half2float(va.data[5]) / __half2float(vb.data[5]);
                float r6 = __half2float(va.data[6]) / __half2float(vb.data[6]);
                float r7 = __half2float(va.data[7]) / __half2float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i], half8_add(v4_load(&output[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < total; i += stride) {
            float rv = s_load(&a[i]) / s_load(&b[i]);
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Min
    // ================================================================
    case EltwiseType::Min: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                float4 va = v4_load(&a[i]);
                float4 vb = v4_load(&b[i]);
                float r0 = fminf(va.x, vb.x);
                float r1 = fminf(va.y, vb.y);
                float r2 = fminf(va.z, vb.z);
                float r3 = fminf(va.w, vb.w);
                if (add_to) {
                    float4 vo = v4_load(&output[i]);
                    v4_store(&output[i], make_float4(
                        vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&output[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                half8 va = v4_load(&a[i]);
                half8 vb = v4_load(&b[i]);
                float r0 = fminf(__half2float(va.data[0]), __half2float(vb.data[0]));
                float r1 = fminf(__half2float(va.data[1]), __half2float(vb.data[1]));
                float r2 = fminf(__half2float(va.data[2]), __half2float(vb.data[2]));
                float r3 = fminf(__half2float(va.data[3]), __half2float(vb.data[3]));
                float r4 = fminf(__half2float(va.data[4]), __half2float(vb.data[4]));
                float r5 = fminf(__half2float(va.data[5]), __half2float(vb.data[5]));
                float r6 = fminf(__half2float(va.data[6]), __half2float(vb.data[6]));
                float r7 = fminf(__half2float(va.data[7]), __half2float(vb.data[7]));
                if (add_to) {
                    v4_store(&output[i], half8_add(v4_load(&output[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < total; i += stride) {
            float av = s_load(&a[i]);
            float bv = s_load(&b[i]);
            float rv = av < bv ? av : bv;
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Max
    // ================================================================
    case EltwiseType::Max: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                float4 va = v4_load(&a[i]);
                float4 vb = v4_load(&b[i]);
                float r0 = fmaxf(va.x, vb.x);
                float r1 = fmaxf(va.y, vb.y);
                float r2 = fmaxf(va.z, vb.z);
                float r3 = fmaxf(va.w, vb.w);
                if (add_to) {
                    float4 vo = v4_load(&output[i]);
                    v4_store(&output[i], make_float4(
                        vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&output[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                half8 va = v4_load(&a[i]);
                half8 vb = v4_load(&b[i]);
                float r0 = fmaxf(__half2float(va.data[0]), __half2float(vb.data[0]));
                float r1 = fmaxf(__half2float(va.data[1]), __half2float(vb.data[1]));
                float r2 = fmaxf(__half2float(va.data[2]), __half2float(vb.data[2]));
                float r3 = fmaxf(__half2float(va.data[3]), __half2float(vb.data[3]));
                float r4 = fmaxf(__half2float(va.data[4]), __half2float(vb.data[4]));
                float r5 = fmaxf(__half2float(va.data[5]), __half2float(vb.data[5]));
                float r6 = fmaxf(__half2float(va.data[6]), __half2float(vb.data[6]));
                float r7 = fmaxf(__half2float(va.data[7]), __half2float(vb.data[7]));
                if (add_to) {
                    v4_store(&output[i], half8_add(v4_load(&output[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < total; i += stride) {
            float av = s_load(&a[i]);
            float bv = s_load(&b[i]);
            float rv = av > bv ? av : bv;
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    // ================================================================
    // Pow
    // ================================================================
    case EltwiseType::Pow: {
        if constexpr (sizeof(T) == 4) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                float4 va = v4_load(&a[i]);
                float4 vb = v4_load(&b[i]);
                float r0 = powf(va.x, vb.x);
                float r1 = powf(va.y, vb.y);
                float r2 = powf(va.z, vb.z);
                float r3 = powf(va.w, vb.w);
                if (add_to) {
                    float4 vo = v4_load(&output[i]);
                    v4_store(&output[i], make_float4(
                        vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&output[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                half8 va = v4_load(&a[i]);
                half8 vb = v4_load(&b[i]);
                float r0 = powf(__half2float(va.data[0]), __half2float(vb.data[0]));
                float r1 = powf(__half2float(va.data[1]), __half2float(vb.data[1]));
                float r2 = powf(__half2float(va.data[2]), __half2float(vb.data[2]));
                float r3 = powf(__half2float(va.data[3]), __half2float(vb.data[3]));
                float r4 = powf(__half2float(va.data[4]), __half2float(vb.data[4]));
                float r5 = powf(__half2float(va.data[5]), __half2float(vb.data[5]));
                float r6 = powf(__half2float(va.data[6]), __half2float(vb.data[6]));
                float r7 = powf(__half2float(va.data[7]), __half2float(vb.data[7]));
                if (add_to) {
                    v4_store(&output[i], half8_add(v4_load(&output[i]),
                        r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], half8_set(r0, r1, r2, r3, r4, r5, r6, r7));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < total; i += stride) {
            float rv = powf(s_load(&a[i]), s_load(&b[i]));
            if (add_to) {
                s_store(&output[i], s_load(&output[i]) + rv);
            } else {
                s_store(&output[i], rv);
            }
        }
        break;
    }
    }
}

// ============================================================
// Host-side launcher
// ============================================================

template <typename T>
void eltwise_cuda_impl(
    const EltwiseAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    const int64_t total = A.numel();
    if (total == 0) return;
    NNOPS_ASSERT(A.numel() == B.numel());
    NNOPS_ASSERT(output.numel() == total);

    const int64_t rank = A.rank();
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    if (rank <= 1) {
        // Flat tensor — use flat kernel with grid-stride loop
        const int block_size = kCudaBlockSize;
        const int grid_size = std::min(
            static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))),
            65535);
        eltwise_flat_kernel<<<grid_size, block_size, 0, stream>>>(
            A.ptr<T>(), B.ptr<T>(), output.ptr<T>(),
            total, attrs.type, attrs.add_to);
    } else {
        // Row-by-row processing, respecting pitch.
        // For packed layouts, last_dim = W * pack covers all C8 lanes,
        // and num_rows = total_rows() accounts for channel block rounding.
        const int64_t last_dim = A.shape(rank - 1) * A.channel_pack_size();
        const int64_t num_rows = A.total_rows();
        const int64_t a_row_stride = A.row_stride_elems();
        const int64_t b_row_stride = B.row_stride_elems();
        const int64_t o_row_stride = output.row_stride_elems();

        const int block_size = std::min(static_cast<int>(last_dim), kCudaBlockSize);

        eltwise_kernel<<<num_rows, block_size, 0, stream>>>(
            A.ptr<T>(), B.ptr<T>(), output.ptr<T>(),
            num_rows, last_dim, a_row_stride, b_row_stride, o_row_stride,
            attrs.type, attrs.add_to);
    }
}

// ============================================================
// Entry point
// ============================================================

void eltwise_cuda(
    const EltwiseAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        eltwise_cuda_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        eltwise_cuda_impl<__half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"eltwise_cuda: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cuda
