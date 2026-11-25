/// @file where.cu
/// @brief CUDA Where — a broadcast select over three operands.
///
/// The condition is read as raw bytes (s8 and u8 share one nonzero test) and
/// the branches are typed. All three go through the broadcast walk, so a mask
/// row or a scalar branch costs one load per output row. When the three
/// operands are all stride 1 the inner loop is four or eight lanes wide; a
/// broadcast operand keeps the scalar walk, where each of its loads is a repeat
/// anyway.
///
/// Branches: f32, f16, s8, u8, s32, s64.

#include "bcast.cuh"
#include "nnops/ops/where.hpp"

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

template <class T>
__global__ void where_kernel(const uint8_t* __restrict__ cond, const T* __restrict__ x,
                             const T* __restrict__ y, T* __restrict__ out, int64_t rows,
                             int64_t cols, BcastOperand pc, BcastOperand px, BcastOperand py)
{
    __shared__ int64_t s_off[3];
    const int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = int64_t(blockDim.x) * gridDim.x;
    for (int64_t r = blockIdx.y; r < rows; r += gridDim.y) {
        const int64_t base = r * cols;
        if (!(pc.linear && px.linear && py.linear)) {
            if (threadIdx.x == 0) {
                s_off[0] = pc.linear ? base : bcast_offset(pc.layout, base);
                s_off[1] = px.linear ? base : bcast_offset(px.layout, base);
                s_off[2] = py.linear ? base : bcast_offset(py.layout, base);
            }
            __syncthreads();
        }
        const uint8_t* crow = cond + (pc.linear ? base : s_off[0]);
        const T* xrow = x + (px.linear ? base : s_off[1]);
        const T* yrow = y + (py.linear ? base : s_off[2]);
        T* orow = out + base;

        constexpr int VEC = std::is_same_v<T, float> ? 4 : (sizeof(T) == 2 ? 8 : 1);
        const bool vec = VEC > 1 && pc.step == 1 && px.step == 1 && py.step == 1 &&
                         aligned16(crow, xrow, yrow) && aligned16(orow);
        const int64_t vec_end = vec ? (cols / VEC) * VEC : 0;

        if constexpr (VEC > 1) {
            for (int64_t i = tid * VEC; i < vec_end; i += stride * VEC) {
                if constexpr (std::is_same_v<T, float>) {
                    const float4 vx = v4_load(xrow + i);
                    const float4 vy = v4_load(yrow + i);
                    const float xs[4] = {vx.x, vx.y, vx.z, vx.w};
                    const float ys[4] = {vy.x, vy.y, vy.z, vy.w};
                    float rs[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        rs[k] = crow[i + k] != 0 ? xs[k] : ys[k];
                    }
                    v4_store(orow + i, make_float4(rs[0], rs[1], rs[2], rs[3]));
                } else {
                    const pack8<T> vx = v4_load(xrow + i);
                    const pack8<T> vy = v4_load(yrow + i);
                    float rs[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        rs[k] = crow[i + k] != 0 ? elem_to_float(vx.data[k])
                                                 : elem_to_float(vy.data[k]);
                    }
                    v4_store(orow + i, pack8_set<T>(rs[0], rs[1], rs[2], rs[3], rs[4], rs[5],
                                                    rs[6], rs[7]));
                }
            }
        }
        for (int64_t i = vec_end + tid; i < cols; i += stride) {
            orow[i] = crow[i * pc.step] != 0 ? xrow[i * px.step] : yrow[i * py.step];
        }
    }
}

template <class T>
Status where_type(TensorView& output, std::span<const TensorView> inputs, cudaStream_t stream)
{
    const TensorView& cond = inputs[0];
    const TensorView& x = inputs[1];
    const TensorView& y = inputs[2];
    const int64_t rank = output.rank();
    if (rank <= 0) {
        return Status::ok;
    }
    int64_t dims[8];
    bcast_out_dims(output, dims);
    const int64_t cols = dims[rank - 1];
    const int64_t rows = cols > 0 ? output.numel() / cols : 0;
    if (rows <= 0 || cols <= 0) {
        return Status::ok;
    }
    const BcastOperand pc = make_bcast(cond, dims, rank);
    const BcastOperand px = make_bcast(x, dims, rank);
    const BcastOperand py = make_bcast(y, dims, rank);
    constexpr int kVec = std::is_same_v<T, float> ? 4 : (sizeof(T) == 2 ? 8 : 1);
    where_kernel<T><<<bcast_grid(cols, rows, kVec), kThreads, 0, stream>>>(
        cond.ptr<uint8_t>(), x.ptr<T>(), y.ptr<T>(), output.ptr<T>(), rows, cols, pc, px, py);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // anonymous namespace

Status where_cuda(const WhereAttributes&, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    // The condition is u8/s8; a signed byte is nonzero exactly when its bits
    // are, so both read through the same pointer.
    switch (inputs[1].data_type()) {
    case DataType::f32:
        return where_type<float>(output, inputs, stream);
    case DataType::f16:
        return where_type<__half>(output, inputs, stream);
    case DataType::bf16:
        return where_type<__nv_bfloat16>(output, inputs, stream);
    case DataType::s8:
        return where_type<int8_t>(output, inputs, stream);
    case DataType::u8:
        return where_type<uint8_t>(output, inputs, stream);
    case DataType::s32:
        return where_type<int32_t>(output, inputs, stream);
    case DataType::s64:
        return where_type<int64_t>(output, inputs, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
