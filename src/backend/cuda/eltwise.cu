/// @file eltwise.cu
/// @brief CUDA implementation of element-wise binary operations.
///
/// Element-wise operations — flat grid-stride loop for rank < 2,
/// row-by-row pitch-aware kernel for rank >= 2 (supports packed layouts).
/// All 4 operations: Add, Sub, Mul, Div.
///
/// Supports f32, f16 and bf16. Workspace: not used.

#include "bcast.cuh"
#include "cuda_common.cuh"
#include "nnops/ops/eltwise.hpp"
#include "nnops/detail/assert.hpp"

namespace nnops::backend::cuda {

// ============================================================
// Row-by-row CUDA kernel — pitch-aware (rank >= 2)
// ============================================================

template <typename T>
__global__ void eltwise_kernel(const T* __restrict__ a, const T* __restrict__ b,
                               T* __restrict__ output, int64_t num_rows, int64_t last_dim,
                               int64_t a_row_stride, int64_t b_row_stride, int64_t o_row_stride,
                               EltwiseType op_type, bool add_to)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4; // 128-bit / sizeof(T)

    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    if (row >= num_rows)
        return;

    const T* a_row = a + row * a_row_stride;
    const T* b_row = b + row * b_row_stride;
    T* o_row = output + row * o_row_stride;

    // An unaligned row walks the scalar loops instead of the 16 B ones.
    const int64_t vec_end = aligned16(a_row, b_row, o_row) ? (last_dim / VEC) * VEC : 0;

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
                    v4_store(&o_row[i], make_float4(vo.x + va.x + vb.x, vo.y + va.y + vb.y,
                                                    vo.z + va.z + vb.z, vo.w + va.w + vb.w));
                } else {
                    v4_store(&o_row[i],
                             make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                pack8<T> va = v4_load(&a_row[i]);
                pack8<T> vb = v4_load(&b_row[i]);
                float r0 = elem_to_float(va.data[0]) + elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) + elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) + elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) + elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) + elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) + elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) + elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) + elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i],
                             pack8_add<T>(v4_load(&o_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&o_row[i], make_float4(vo.x + va.x - vb.x, vo.y + va.y - vb.y,
                                                    vo.z + va.z - vb.z, vo.w + va.w - vb.w));
                } else {
                    v4_store(&o_row[i],
                             make_float4(va.x - vb.x, va.y - vb.y, va.z - vb.z, va.w - vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                pack8<T> va = v4_load(&a_row[i]);
                pack8<T> vb = v4_load(&b_row[i]);
                float r0 = elem_to_float(va.data[0]) - elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) - elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) - elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) - elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) - elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) - elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) - elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) - elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i],
                             pack8_add<T>(v4_load(&o_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&o_row[i], make_float4(vo.x + va.x * vb.x, vo.y + va.y * vb.y,
                                                    vo.z + va.z * vb.z, vo.w + va.w * vb.w));
                } else {
                    v4_store(&o_row[i],
                             make_float4(va.x * vb.x, va.y * vb.y, va.z * vb.z, va.w * vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                pack8<T> va = v4_load(&a_row[i]);
                pack8<T> vb = v4_load(&b_row[i]);
                float r0 = elem_to_float(va.data[0]) * elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) * elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) * elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) * elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) * elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) * elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) * elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) * elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i],
                             pack8_add<T>(v4_load(&o_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&o_row[i], make_float4(vo.x + va.x / vb.x, vo.y + va.y / vb.y,
                                                    vo.z + va.z / vb.z, vo.w + va.w / vb.w));
                } else {
                    v4_store(&o_row[i],
                             make_float4(va.x / vb.x, va.y / vb.y, va.z / vb.z, va.w / vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                pack8<T> va = v4_load(&a_row[i]);
                pack8<T> vb = v4_load(&b_row[i]);
                float r0 = elem_to_float(va.data[0]) / elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) / elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) / elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) / elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) / elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) / elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) / elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) / elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&o_row[i],
                             pack8_add<T>(v4_load(&o_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&o_row[i], make_float4(vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&o_row[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                pack8<T> va = v4_load(&a_row[i]);
                pack8<T> vb = v4_load(&b_row[i]);
                float r0 = fminf(elem_to_float(va.data[0]), elem_to_float(vb.data[0]));
                float r1 = fminf(elem_to_float(va.data[1]), elem_to_float(vb.data[1]));
                float r2 = fminf(elem_to_float(va.data[2]), elem_to_float(vb.data[2]));
                float r3 = fminf(elem_to_float(va.data[3]), elem_to_float(vb.data[3]));
                float r4 = fminf(elem_to_float(va.data[4]), elem_to_float(vb.data[4]));
                float r5 = fminf(elem_to_float(va.data[5]), elem_to_float(vb.data[5]));
                float r6 = fminf(elem_to_float(va.data[6]), elem_to_float(vb.data[6]));
                float r7 = fminf(elem_to_float(va.data[7]), elem_to_float(vb.data[7]));
                if (add_to) {
                    v4_store(&o_row[i],
                             pack8_add<T>(v4_load(&o_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&o_row[i], make_float4(vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&o_row[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                pack8<T> va = v4_load(&a_row[i]);
                pack8<T> vb = v4_load(&b_row[i]);
                float r0 = fmaxf(elem_to_float(va.data[0]), elem_to_float(vb.data[0]));
                float r1 = fmaxf(elem_to_float(va.data[1]), elem_to_float(vb.data[1]));
                float r2 = fmaxf(elem_to_float(va.data[2]), elem_to_float(vb.data[2]));
                float r3 = fmaxf(elem_to_float(va.data[3]), elem_to_float(vb.data[3]));
                float r4 = fmaxf(elem_to_float(va.data[4]), elem_to_float(vb.data[4]));
                float r5 = fmaxf(elem_to_float(va.data[5]), elem_to_float(vb.data[5]));
                float r6 = fmaxf(elem_to_float(va.data[6]), elem_to_float(vb.data[6]));
                float r7 = fmaxf(elem_to_float(va.data[7]), elem_to_float(vb.data[7]));
                if (add_to) {
                    v4_store(&o_row[i],
                             pack8_add<T>(v4_load(&o_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&o_row[i], make_float4(vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&o_row[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += blockDim.x * VEC) {
                pack8<T> va = v4_load(&a_row[i]);
                pack8<T> vb = v4_load(&b_row[i]);
                float r0 = powf(elem_to_float(va.data[0]), elem_to_float(vb.data[0]));
                float r1 = powf(elem_to_float(va.data[1]), elem_to_float(vb.data[1]));
                float r2 = powf(elem_to_float(va.data[2]), elem_to_float(vb.data[2]));
                float r3 = powf(elem_to_float(va.data[3]), elem_to_float(vb.data[3]));
                float r4 = powf(elem_to_float(va.data[4]), elem_to_float(vb.data[4]));
                float r5 = powf(elem_to_float(va.data[5]), elem_to_float(vb.data[5]));
                float r6 = powf(elem_to_float(va.data[6]), elem_to_float(vb.data[6]));
                float r7 = powf(elem_to_float(va.data[7]), elem_to_float(vb.data[7]));
                if (add_to) {
                    v4_store(&o_row[i],
                             pack8_add<T>(v4_load(&o_row[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&o_row[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
// Broadcast kernel — operands of different shapes
// ============================================================

/// One element of the operator, in float. Every path here works in float for a
/// 16-bit element type, the same as the pair of kernels above.
__device__ __forceinline__ float eltwise_apply_f(EltwiseType t, float av, float bv)
{
    switch (t) {
    case EltwiseType::Add:
        return av + bv;
    case EltwiseType::Sub:
        return av - bv;
    case EltwiseType::Mul:
        return av * bv;
    case EltwiseType::Div:
        return av / bv;
    case EltwiseType::Min:
        return av < bv ? av : bv;
    case EltwiseType::Max:
        return av > bv ? av : bv;
    case EltwiseType::Pow:
        return powf(av, bv);
    }
    return av;
}

/// The broadcast walk `bcast.cuh` shares with Compare, Logical and Where: the
/// output is flat, each operand's offset for an output row comes from one
/// mixed-radix decode on thread 0, and inside the row the operand advances by
/// its last-axis step -- 0 when that axis was broadcast. Before this the pair
/// of kernels above assumed both operands had the output's shape, so a bias add
/// of [N,C] against [C] read past the operand.
template <class T>
__global__ void eltwise_bcast_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                     T* __restrict__ out, int64_t rows, int64_t cols,
                                     int64_t o_row_stride, BcastOperand pa, BcastOperand pb,
                                     EltwiseType op_type, bool add_to)
{
    __shared__ int64_t s_off[2];
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const int64_t base = r * cols;
        if (!(pa.linear && pb.linear)) {
            if (threadIdx.x == 0) {
                s_off[0] = pa.linear ? base : bcast_offset(pa.layout, base);
                s_off[1] = pb.linear ? base : bcast_offset(pb.layout, base);
            }
            __syncthreads();
        }
        const T* const arow = a + (pa.linear ? base : s_off[0]);
        const T* const brow = b + (pb.linear ? base : s_off[1]);
        // The output is walked by its own pitch: a row shorter than the 32 B
        // alignment is padded, and the flat index is not the address there.
        T* const orow = out + r * o_row_stride;
        for (int64_t i = tid; i < cols; i += stride) {
            float rv = eltwise_apply_f(op_type, s_load(arow + i * pa.step), s_load(brow + i * pb.step));
            if (add_to) {
                rv += s_load(orow + i);
            }
            s_store(orow + i, rv);
        }
    }
}

// ============================================================
// Flat kernel for 1D / scalar tensors
// ============================================================

template <typename T>
__global__ void eltwise_flat_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                    T* __restrict__ output, int64_t total, EltwiseType op_type,
                                    bool add_to)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4; // 128-bit / sizeof(T)

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
                    v4_store(&output[i], make_float4(vo.x + va.x + vb.x, vo.y + va.y + vb.y,
                                                     vo.z + va.z + vb.z, vo.w + va.w + vb.w));
                } else {
                    v4_store(&output[i],
                             make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                pack8<T> va = v4_load(&a[i]);
                pack8<T> vb = v4_load(&b[i]);
                float r0 = elem_to_float(va.data[0]) + elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) + elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) + elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) + elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) + elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) + elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) + elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) + elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i],
                             pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&output[i], make_float4(vo.x + va.x - vb.x, vo.y + va.y - vb.y,
                                                     vo.z + va.z - vb.z, vo.w + va.w - vb.w));
                } else {
                    v4_store(&output[i],
                             make_float4(va.x - vb.x, va.y - vb.y, va.z - vb.z, va.w - vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                pack8<T> va = v4_load(&a[i]);
                pack8<T> vb = v4_load(&b[i]);
                float r0 = elem_to_float(va.data[0]) - elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) - elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) - elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) - elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) - elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) - elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) - elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) - elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i],
                             pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&output[i], make_float4(vo.x + va.x * vb.x, vo.y + va.y * vb.y,
                                                     vo.z + va.z * vb.z, vo.w + va.w * vb.w));
                } else {
                    v4_store(&output[i],
                             make_float4(va.x * vb.x, va.y * vb.y, va.z * vb.z, va.w * vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                pack8<T> va = v4_load(&a[i]);
                pack8<T> vb = v4_load(&b[i]);
                float r0 = elem_to_float(va.data[0]) * elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) * elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) * elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) * elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) * elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) * elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) * elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) * elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i],
                             pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&output[i], make_float4(vo.x + va.x / vb.x, vo.y + va.y / vb.y,
                                                     vo.z + va.z / vb.z, vo.w + va.w / vb.w));
                } else {
                    v4_store(&output[i],
                             make_float4(va.x / vb.x, va.y / vb.y, va.z / vb.z, va.w / vb.w));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                pack8<T> va = v4_load(&a[i]);
                pack8<T> vb = v4_load(&b[i]);
                float r0 = elem_to_float(va.data[0]) / elem_to_float(vb.data[0]);
                float r1 = elem_to_float(va.data[1]) / elem_to_float(vb.data[1]);
                float r2 = elem_to_float(va.data[2]) / elem_to_float(vb.data[2]);
                float r3 = elem_to_float(va.data[3]) / elem_to_float(vb.data[3]);
                float r4 = elem_to_float(va.data[4]) / elem_to_float(vb.data[4]);
                float r5 = elem_to_float(va.data[5]) / elem_to_float(vb.data[5]);
                float r6 = elem_to_float(va.data[6]) / elem_to_float(vb.data[6]);
                float r7 = elem_to_float(va.data[7]) / elem_to_float(vb.data[7]);
                if (add_to) {
                    v4_store(&output[i],
                             pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&output[i], make_float4(vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&output[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                pack8<T> va = v4_load(&a[i]);
                pack8<T> vb = v4_load(&b[i]);
                float r0 = fminf(elem_to_float(va.data[0]), elem_to_float(vb.data[0]));
                float r1 = fminf(elem_to_float(va.data[1]), elem_to_float(vb.data[1]));
                float r2 = fminf(elem_to_float(va.data[2]), elem_to_float(vb.data[2]));
                float r3 = fminf(elem_to_float(va.data[3]), elem_to_float(vb.data[3]));
                float r4 = fminf(elem_to_float(va.data[4]), elem_to_float(vb.data[4]));
                float r5 = fminf(elem_to_float(va.data[5]), elem_to_float(vb.data[5]));
                float r6 = fminf(elem_to_float(va.data[6]), elem_to_float(vb.data[6]));
                float r7 = fminf(elem_to_float(va.data[7]), elem_to_float(vb.data[7]));
                if (add_to) {
                    v4_store(&output[i],
                             pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&output[i], make_float4(vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&output[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                pack8<T> va = v4_load(&a[i]);
                pack8<T> vb = v4_load(&b[i]);
                float r0 = fmaxf(elem_to_float(va.data[0]), elem_to_float(vb.data[0]));
                float r1 = fmaxf(elem_to_float(va.data[1]), elem_to_float(vb.data[1]));
                float r2 = fmaxf(elem_to_float(va.data[2]), elem_to_float(vb.data[2]));
                float r3 = fmaxf(elem_to_float(va.data[3]), elem_to_float(vb.data[3]));
                float r4 = fmaxf(elem_to_float(va.data[4]), elem_to_float(vb.data[4]));
                float r5 = fmaxf(elem_to_float(va.data[5]), elem_to_float(vb.data[5]));
                float r6 = fmaxf(elem_to_float(va.data[6]), elem_to_float(vb.data[6]));
                float r7 = fmaxf(elem_to_float(va.data[7]), elem_to_float(vb.data[7]));
                if (add_to) {
                    v4_store(&output[i],
                             pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
                    v4_store(&output[i], make_float4(vo.x + r0, vo.y + r1, vo.z + r2, vo.w + r3));
                } else {
                    v4_store(&output[i], make_float4(r0, r1, r2, r3));
                }
            }
        } else {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                pack8<T> va = v4_load(&a[i]);
                pack8<T> vb = v4_load(&b[i]);
                float r0 = powf(elem_to_float(va.data[0]), elem_to_float(vb.data[0]));
                float r1 = powf(elem_to_float(va.data[1]), elem_to_float(vb.data[1]));
                float r2 = powf(elem_to_float(va.data[2]), elem_to_float(vb.data[2]));
                float r3 = powf(elem_to_float(va.data[3]), elem_to_float(vb.data[3]));
                float r4 = powf(elem_to_float(va.data[4]), elem_to_float(vb.data[4]));
                float r5 = powf(elem_to_float(va.data[5]), elem_to_float(vb.data[5]));
                float r6 = powf(elem_to_float(va.data[6]), elem_to_float(vb.data[6]));
                float r7 = powf(elem_to_float(va.data[7]), elem_to_float(vb.data[7]));
                if (add_to) {
                    v4_store(&output[i],
                             pack8_add<T>(v4_load(&output[i]), r0, r1, r2, r3, r4, r5, r6, r7));
                } else {
                    v4_store(&output[i], pack8_set<T>(r0, r1, r2, r3, r4, r5, r6, r7));
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
void eltwise_cuda_impl(const EltwiseAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& A = inputs[0];
    const auto& B = inputs[1];
    if (output.numel() == 0)
        return;
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);

    // The broadcast form first: an operand whose shape is not the output's
    // takes the row walk below. Two dense operands of the output's own shape
    // resolve to `linear` and keep the pair of kernels this file has always
    // used, which are vectorized and unrolled per operator.
    int64_t dims[8];
    bcast_out_dims(output, dims);
    const int64_t orank = output.rank();
    const BcastOperand pa = make_bcast(A, dims, orank);
    const BcastOperand pb = make_bcast(B, dims, orank);
    if (!(pa.linear && pb.linear)) {
        const int64_t cols = dims[orank - 1];
        const int64_t rows = cols > 0 ? output.numel() / cols : 0;
        if (rows <= 0 || cols <= 0)
            return;
        const dim3 grid = bcast_grid(cols, rows, 1);
        eltwise_bcast_kernel<T><<<grid, kBcastThreads, 0, stream>>>(
            A.ptr<T>(), B.ptr<T>(), output.ptr<T>(), rows, cols, output.row_stride_elems(), pa,
            pb, attrs.type, attrs.add_to);
        return;
    }

    const int64_t total = A.numel();
    const int64_t rank = A.rank();

    if (rank <= 1) {
        // Flat tensor — use flat kernel with grid-stride loop
        const int block_size = kCudaBlockSize;
        const int grid_size =
            std::min(static_cast<int>(ceil_div(total, static_cast<int64_t>(block_size))), 65535);
        eltwise_flat_kernel<<<grid_size, block_size, 0, stream>>>(
            A.ptr<T>(), B.ptr<T>(), output.ptr<T>(), total, attrs.type, attrs.add_to);
    } else {
        // Row-by-row processing, respecting pitch. An NHWC row is one
        // spatial line — W pixels × C channels.
        const int64_t last_dim = (rank >= 3 && is_channel_last(A.layout())) ?
                                     A.shape(rank - 2) * A.shape(rank - 1) :
                                     A.shape(rank - 1);
        const int64_t num_rows = A.total_rows();
        const int64_t a_row_stride = A.row_stride_elems();
        const int64_t b_row_stride = B.row_stride_elems();
        const int64_t o_row_stride = output.row_stride_elems();

        const int block_size = std::min(static_cast<int>(last_dim), kCudaBlockSize);

        eltwise_kernel<<<num_rows, block_size, 0, stream>>>(
            A.ptr<T>(), B.ptr<T>(), output.ptr<T>(), num_rows, last_dim, a_row_stride, b_row_stride,
            o_row_stride, attrs.type, attrs.add_to);
    }
}

// ============================================================
// Entry point
// ============================================================

Status eltwise_cuda(const EltwiseAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx,
                    void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        eltwise_cuda_impl<float>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::f16:
        eltwise_cuda_impl<__half>(attrs, output, inputs, ctx);
        return Status::ok;
    case DataType::bf16:
        eltwise_cuda_impl<__nv_bfloat16>(attrs, output, inputs, ctx);
        return Status::ok;
    default:
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cuda
