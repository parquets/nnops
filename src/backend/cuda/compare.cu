/// @file compare.cu
/// @brief CUDA Compare and Logical — broadcast to u8.
///
/// Both walk the broadcast engine (bcast.cuh): one block slice per output row,
/// the operand's row offset decoded once, a constant stride inside the row
/// (zero when that axis was broadcast). Compare reads a 2-byte float through
/// the widening conversion, so the relation runs on the value and NaN compares
/// unequal to everything; the integer operands keep their own width, so an s64
/// relation stays exact. Logical reads the raw bytes/words: nonzero is true.
///
/// The stride-1 pair — the usual same-shape case — takes a four- or eight-lane
/// inner loop; a broadcast run stays scalar, where each element's operand is a
/// repeat of one load anyway.
///
/// Compare: f32, f16, bf16, s32, s64. Logical: s8, u8, s32, s64.

#include "bcast.cuh"
#include "nnops/ops/compare.hpp"

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

/// The comparison value: floats widen, integers keep their exact width.
template <class T> __device__ inline auto cmp_val(T v) noexcept
{
    if constexpr (std::is_integral_v<T>) {
        return static_cast<int64_t>(v);
    } else {
        return elem_to_float(v);
    }
}

template <int REL, class V> __device__ inline bool relation(V a, V b) noexcept
{
    if constexpr (REL == 0) {
        return a == b;
    } else if constexpr (REL == 1) {
        return a > b;
    } else if constexpr (REL == 2) {
        return a < b;
    } else if constexpr (REL == 3) {
        return a >= b;
    } else {
        return a <= b;
    }
}

template <class T, int REL>
__global__ void compare_kernel(const T* __restrict__ a, const T* __restrict__ b,
                               uint8_t* __restrict__ out, int64_t rows, int64_t cols,
                               BcastOperand pa, BcastOperand pb)
{
    __shared__ int64_t s_off[2];
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const int64_t base = r * cols;
        // A dense operand's offset is the flat index. A broadcast one decodes
        // once per row on thread 0 and shares it: the odometer is 16 int64 of
        // layout, which is far too much state for every thread of a thin row.
        if (!(pa.linear && pb.linear)) {
            if (threadIdx.x == 0) {
                s_off[0] = pa.linear ? base : bcast_offset(pa.layout, base);
                s_off[1] = pb.linear ? base : bcast_offset(pb.layout, base);
            }
            __syncthreads();
        }
        const T* arow = a + (pa.linear ? base : s_off[0]);
        const T* brow = b + (pb.linear ? base : s_off[1]);
        uint8_t* orow = out + base;

        constexpr int VEC = std::is_same_v<T, float> ? 4 : (sizeof(T) == 2 ? 8 : 1);
        const bool vec = VEC > 1 && pa.step == 1 && pb.step == 1 &&
                         aligned16(arow, brow, orow);
        const int64_t vec_end = vec ? (cols / VEC) * VEC : 0;

        if constexpr (VEC > 1) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                if constexpr (std::is_same_v<T, float>) {
                    const float4 va = v4_load(arow + i);
                    const float4 vb = v4_load(brow + i);
                    const uint32_t packed =
                        uint32_t(relation<REL>(cmp_val(va.x), cmp_val(vb.x)) ? 1 : 0) |
                        (uint32_t(relation<REL>(cmp_val(va.y), cmp_val(vb.y)) ? 1 : 0) << 8) |
                        (uint32_t(relation<REL>(cmp_val(va.z), cmp_val(vb.z)) ? 1 : 0) << 16) |
                        (uint32_t(relation<REL>(cmp_val(va.w), cmp_val(vb.w)) ? 1 : 0) << 24);
                    *reinterpret_cast<uint32_t*>(orow + i) = packed;
                } else {
                    const pack8<T> va = v4_load(arow + i);
                    const pack8<T> vb = v4_load(brow + i);
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        orow[i + k] =
                            relation<REL>(cmp_val(va.data[k]), cmp_val(vb.data[k]));
                    }
                }
            }
        }
        for (int64_t i = vec_end + tid; i < cols; i += stride) {
            orow[i] = relation<REL>(cmp_val(arow[i * pa.step]), cmp_val(brow[i * pb.step]));
        }
    }
}

template <class T, int REL>
Status compare_run(const TensorView& a, const TensorView& b, TensorView& out, int64_t rank,
                   cudaStream_t stream)
{
    int64_t dims[8];
    bcast_out_dims(out, dims);
    const BcastOperand pa = make_bcast(a, dims, rank);
    const BcastOperand pb = make_bcast(b, dims, rank);
    const int64_t cols = dims[rank - 1];
    const int64_t rows = cols > 0 ? out.numel() / cols : 0;
    if (rows <= 0 || cols <= 0) {
        return Status::ok;
    }
    constexpr int kVec = std::is_same_v<T, float> ? 4 : (sizeof(T) == 2 ? 8 : 1);
    const dim3 grid = bcast_grid(cols, rows, kVec);
    compare_kernel<T, REL><<<grid, kThreads, 0, stream>>>(
        a.ptr<T>(), b.ptr<T>(), out.ptr<uint8_t>(), rows, cols, pa, pb);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

template <class T>
Status compare_type(CompareType type, const TensorView& a, const TensorView& b, TensorView& out,
                    int64_t rank, cudaStream_t stream)
{
    switch (type) {
    case CompareType::Equal:
        return compare_run<T, 0>(a, b, out, rank, stream);
    case CompareType::Greater:
        return compare_run<T, 1>(a, b, out, rank, stream);
    case CompareType::Less:
        return compare_run<T, 2>(a, b, out, rank, stream);
    case CompareType::GreaterOrEqual:
        return compare_run<T, 3>(a, b, out, rank, stream);
    case CompareType::LessOrEqual:
        return compare_run<T, 4>(a, b, out, rank, stream);
    }
    return Status::unsupported;
}

template <class T, int OP>
__global__ void logical_binary_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                      uint8_t* __restrict__ out, int64_t rows, int64_t cols,
                                      BcastOperand pa, BcastOperand pb)
{
    __shared__ int64_t s_off[2];
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const int64_t base = r * cols;
        // A dense operand's offset is the flat index. A broadcast one decodes
        // once per row on thread 0 and shares it: the odometer is 16 int64 of
        // layout, which is far too much state for every thread of a thin row.
        if (!(pa.linear && pb.linear)) {
            if (threadIdx.x == 0) {
                s_off[0] = pa.linear ? base : bcast_offset(pa.layout, base);
                s_off[1] = pb.linear ? base : bcast_offset(pb.layout, base);
            }
            __syncthreads();
        }
        const T* arow = a + (pa.linear ? base : s_off[0]);
        const T* brow = b + (pb.linear ? base : s_off[1]);
        uint8_t* orow = out + base;
        for (int64_t i = tid; i < cols; i += stride) {
            const bool x = arow[i * pa.step] != 0;
            const bool y = brow[i * pb.step] != 0;
            if constexpr (OP == 0) {
                orow[i] = (x && y) ? 1 : 0;
            } else if constexpr (OP == 1) {
                orow[i] = (x || y) ? 1 : 0;
            } else {
                orow[i] = (x != y) ? 1 : 0;
            }
        }
    }
}

template <class T>
__global__ void logical_not_kernel(const T* __restrict__ a, uint8_t* __restrict__ out,
                                   int64_t rows, int64_t cols, BcastOperand pa)
{
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const int64_t base = r * cols;
        const T* arow = a + (pa.linear ? base : bcast_offset(pa.layout, base));
        uint8_t* orow = out + base;
        for (int64_t i = tid; i < cols; i += stride) {
            orow[i] = arow[i * pa.step] == 0 ? 1 : 0;
        }
    }
}

template <class T>
Status logical_binary_type(LogicalType type, const TensorView& a, const TensorView& b,
                           TensorView& out, int64_t rank, cudaStream_t stream)
{
    int64_t dims[8];
    bcast_out_dims(out, dims);
    const BcastOperand pa = make_bcast(a, dims, rank);
    const BcastOperand pb = make_bcast(b, dims, rank);
    const int64_t cols = dims[rank - 1];
    const int64_t rows = cols > 0 ? out.numel() / cols : 0;
    if (rows <= 0 || cols <= 0) {
        return Status::ok;
    }
    const dim3 grid = bcast_grid(cols, rows, 1);
    switch (type) {
    case LogicalType::And:
        logical_binary_kernel<T, 0><<<grid, kThreads, 0, stream>>>(
            a.ptr<T>(), b.ptr<T>(), out.ptr<uint8_t>(), rows, cols, pa, pb);
        break;
    case LogicalType::Or:
        logical_binary_kernel<T, 1><<<grid, kThreads, 0, stream>>>(
            a.ptr<T>(), b.ptr<T>(), out.ptr<uint8_t>(), rows, cols, pa, pb);
        break;
    case LogicalType::Xor:
        logical_binary_kernel<T, 2><<<grid, kThreads, 0, stream>>>(
            a.ptr<T>(), b.ptr<T>(), out.ptr<uint8_t>(), rows, cols, pa, pb);
        break;
    default:
        return Status::unsupported;
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

template <class T>
Status logical_not_type(const TensorView& a, TensorView& out, int64_t rank, cudaStream_t stream)
{
    int64_t dims[8];
    bcast_out_dims(out, dims);
    const BcastOperand pa = make_bcast(a, dims, rank);
    const int64_t cols = dims[rank - 1];
    const int64_t rows = cols > 0 ? out.numel() / cols : 0;
    if (rows <= 0 || cols <= 0) {
        return Status::ok;
    }
    logical_not_kernel<T><<<bcast_grid(cols, rows, 1), kThreads, 0, stream>>>(
        a.ptr<T>(), out.ptr<uint8_t>(), rows, cols, pa);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // anonymous namespace

Status compare_cuda(const CompareAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const int64_t rank = inputs[0].rank() > inputs[1].rank() ? inputs[0].rank()
                                                             : inputs[1].rank();
    if (rank <= 0) {
        return Status::ok;
    }
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return compare_type<float>(attrs.type, inputs[0], inputs[1], output, rank, stream);
    case DataType::f16:
        return compare_type<__half>(attrs.type, inputs[0], inputs[1], output, rank, stream);
    case DataType::bf16:
        return compare_type<__nv_bfloat16>(attrs.type, inputs[0], inputs[1], output, rank,
                                           stream);
    case DataType::s32:
        return compare_type<int32_t>(attrs.type, inputs[0], inputs[1], output, rank, stream);
    case DataType::s64:
        return compare_type<int64_t>(attrs.type, inputs[0], inputs[1], output, rank, stream);
    default:
        return Status::unsupported;
    }
}

Status logical_cuda(const LogicalAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    if (attrs.type == LogicalType::Not) {
        const int64_t rank = inputs[0].rank();
        if (rank <= 0) {
            return Status::ok;
        }
        switch (inputs[0].data_type()) {
        case DataType::s8:
            return logical_not_type<int8_t>(inputs[0], output, rank, stream);
        case DataType::u8:
            return logical_not_type<uint8_t>(inputs[0], output, rank, stream);
        case DataType::s32:
            return logical_not_type<int32_t>(inputs[0], output, rank, stream);
        case DataType::s64:
            return logical_not_type<int64_t>(inputs[0], output, rank, stream);
        default:
            return Status::unsupported;
        }
    }
    const int64_t rank = inputs[0].rank() > inputs[1].rank() ? inputs[0].rank()
                                                             : inputs[1].rank();
    if (rank <= 0) {
        return Status::ok;
    }
    switch (inputs[0].data_type()) {
    case DataType::s8:
        return logical_binary_type<int8_t>(attrs.type, inputs[0], inputs[1], output, rank,
                                           stream);
    case DataType::u8:
        return logical_binary_type<uint8_t>(attrs.type, inputs[0], inputs[1], output, rank,
                                            stream);
    case DataType::s32:
        return logical_binary_type<int32_t>(attrs.type, inputs[0], inputs[1], output, rank,
                                            stream);
    case DataType::s64:
        return logical_binary_type<int64_t>(attrs.type, inputs[0], inputs[1], output, rank,
                                            stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
