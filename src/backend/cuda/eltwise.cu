/// @file eltwise.cu
/// @brief CUDA implementation of element-wise binary operations.
///
/// Element-wise operations — flat grid-stride loop for rank < 2,
/// row-by-row pitch-aware kernel for rank >= 2 (supports packed layouts).
/// All 4 operations: Add, Sub, Mul, Div.
///
/// f32, f16 and bf16 run the kernels above. The quantized route takes s8/u8
/// inputs and output and fuses both dequantizations and the requantization into
/// one pass, walking dense rows of the last dim as the CPU's fused transform
/// does. It requires the two inputs to have the output's shape — a broadcast
/// would need the index arithmetic re-derived per scale row — and refuses
/// add_to, both matching the CPU.
///
/// Workspace: not used.

#include "bcast.cuh"
#include "cuda_common.cuh"
#include "quant_elementwise.cuh"
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
// Quantized route — dequantize both, apply, requantize in one pass
// ============================================================

__device__ inline float eltwise_quant_op(EltwiseType type, float a, float b) noexcept
{
    switch (type) {
    case EltwiseType::Add:
        return a + b;
    case EltwiseType::Sub:
        return a - b;
    case EltwiseType::Mul:
        return a * b;
    case EltwiseType::Div:
        return a / b;
    case EltwiseType::Min:
        return fminf(a, b);
    case EltwiseType::Max:
        return fmaxf(a, b);
    case EltwiseType::Pow:
        return powf(a, b);
    }
    return a;
}

template <class InT, class OutT>
__global__ void eltwise_quant_kernel(const InT* __restrict__ a, const InT* __restrict__ b,
                                     OutT* __restrict__ output, int64_t rows, int64_t cols,
                                     qelem::ParamRef qa, qelem::ParamRef qb,
                                     qelem::ParamRef qout, EltwiseType type)
{
    constexpr int VEC = 16;

    for (int64_t row = blockIdx.y; row < rows; row += gridDim.y) {
        const qelem::RowParam pa = qelem::row_param(qa, row);
        const qelem::RowParam pb = qelem::row_param(qb, row);
        const qelem::RowParam po = qelem::row_param(qout, row);
        const InT* a_row = a + row * cols;
        const InT* b_row = b + row * cols;
        OutT* o_row = output + row * cols;

        const int64_t vec_end =
            aligned16(a_row, b_row, o_row) ? (cols / VEC) * VEC : 0;
        const int64_t vstep = int64_t(gridDim.x) * blockDim.x * VEC;
        for (int64_t i = int64_t(blockIdx.x) * blockDim.x * VEC + threadIdx.x * VEC; i < vec_end;
             i += vstep) {
            const qelem::pack16<InT> va = *reinterpret_cast<const qelem::pack16<InT>*>(&a_row[i]);
            const qelem::pack16<InT> vb = *reinterpret_cast<const qelem::pack16<InT>*>(&b_row[i]);
            qelem::pack16<OutT> r;
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                r.data[k] = qelem::requant<OutT>(
                    eltwise_quant_op(type, qelem::dequant(va.data[k], pa),
                                     qelem::dequant(vb.data[k], pb)),
                    po);
            }
            *reinterpret_cast<qelem::pack16<OutT>*>(&o_row[i]) = r;
        }

        const int64_t sstep = int64_t(gridDim.x) * blockDim.x;
        for (int64_t i = vec_end + int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < cols;
             i += sstep) {
            o_row[i] = qelem::requant<OutT>(
                eltwise_quant_op(type, qelem::dequant(a_row[i], pa), qelem::dequant(b_row[i], pb)),
                po);
        }
    }
}

template <class InT, class OutT>
Status eltwise_quant_cuda(const EltwiseAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, cudaStream_t stream)
{
    if (attrs.add_to) {
        return Status::unsupported;
    }
    const TensorView& a = inputs[0];
    const TensorView& b = inputs[1];
    if (a.numel() != b.numel() || a.numel() != output.numel()) {
        return Status::unsupported;
    }
    const int64_t total = a.numel();
    if (total == 0) {
        return Status::ok;
    }
    const int64_t cols = a.shape(a.rank() - 1);
    if (cols <= 0) {
        return Status::invalid_argument;
    }
    const int64_t rows = total / cols;

    qelem::Stage a_s, a_z, b_s, b_z, o_s, o_z;
    qelem::ParamRef qa, qb, qo;
    Status st = qelem::stage_qparams(a.quant_params(), rows, stream, a_s, a_z, qa);
    if (is_ok(st)) {
        st = qelem::stage_qparams(b.quant_params(), rows, stream, b_s, b_z, qb);
    }
    if (is_ok(st)) {
        st = qelem::stage_qparams(output.quant_params(), rows, stream, o_s, o_z, qo);
    }
    if (is_ok(st)) {
        constexpr int VEC = 16;
        const int block_size = kCudaBlockSize;
        const int64_t col_tiles = ceil_div(cols, int64_t(block_size) * VEC);
        const dim3 grid(static_cast<unsigned>(std::min<int64_t>(col_tiles, 65535)),
                        static_cast<unsigned>(std::min<int64_t>(rows, 65535)));
        eltwise_quant_kernel<InT, OutT><<<grid, block_size, 0, stream>>>(
            a.ptr<InT>(), b.ptr<InT>(), output.ptr<OutT>(), rows, cols, qa, qb, qo, attrs.type);
        st = cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    }
    a_s.release(stream);
    a_z.release(stream);
    b_s.release(stream);
    b_z.release(stream);
    o_s.release(stream);
    o_z.release(stream);
    return st;
}

template <class InT>
Status eltwise_quant_pair(const EltwiseAttributes& attrs, TensorView& output,
                          std::span<const TensorView> inputs, cudaStream_t stream)
{
    if (output.data_type() == DataType::s8) {
        return eltwise_quant_cuda<InT, int8_t>(attrs, output, inputs, stream);
    }
    return eltwise_quant_cuda<InT, uint8_t>(attrs, output, inputs, stream);
}

// ============================================================
// Entry point
// ============================================================

Status eltwise_cuda(const EltwiseAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx,
                    void* /*workspace*/)
{
    const DataType dtype = inputs[0].data_type();
    const DataType out_dtype = output.data_type();
    const bool in_int = dtype == DataType::s8 || dtype == DataType::u8;
    const bool out_int = out_dtype == DataType::s8 || out_dtype == DataType::u8;
    if (in_int || out_int) {
        // Both inputs and the output must be s8/u8 (no float↔int mixing).
        if (!in_int || !out_int || inputs[1].data_type() != dtype) {
            return Status::invalid_argument;
        }
        const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
        if (dtype == DataType::s8) {
            return eltwise_quant_pair<int8_t>(attrs, output, inputs, stream);
        }
        return eltwise_quant_pair<uint8_t>(attrs, output, inputs, stream);
    }
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
