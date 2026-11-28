/// @file rope.cu
/// @brief CUDA RoPE — rotary embedding with mRoPE, position offsets and the
///        long-context scaling modes, plus the fused Q/K RMSNorm.
///
/// The angle table is built once for all (batch block, position, pair) and then
/// reused by every head, which is the split the CPU route makes: the table is
/// the expensive `pow`/`cos`/`sin` work and it does not depend on the head. The
/// per-pair constants (base ratio and the scaling ramp/blend) are computed on
/// the host exactly as the CPU computes them — they depend only on the
/// attributes, the rotary width and the run's furthest start — and travel to
/// the table kernel as four floats a pair.
///
/// One block per (outer, position) row then does the rotation, with the block
/// spanning head_dim. When the fused norm is requested the block first reduces
/// the head's sum of squares, writes the RMS-normalized row into the output and
/// rotates that in place — one pass fewer than norm-then-rope.
///
/// A caller-supplied cos/sin pair replaces the table: it is used in row order,
/// widened once on the device when it arrives in the activation dtype.
///
/// f32, f16 and bf16.

#include "cuda_common.cuh"
#include "backend/common/rope_plan.hpp"
#include "nnops/ops/rope.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 128;
// ---- device -------------------------------------------------------------

/// The table: one row per (block, position), `half` pairs each.
__global__ void rope_table_kernel(float* __restrict__ cos_t, float* __restrict__ sin_t,
                                  const float4* __restrict__ pairs, const int64_t* __restrict__ starts,
                                  int64_t rows, int64_t seq_len, int64_t half, int linear,
                                  float factor, float magnitude, int yarn, int llama3)
{
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * half;
         i += int64_t(blockDim.x) * gridDim.x) {
        const int64_t row = i / half;
        const int64_t p = i - row * half;
        const int64_t blk = row / seq_len;
        const int64_t r = row - blk * seq_len;
        const float pos = static_cast<float>(starts[blk] + r);
        const float4 pp = pairs[p];
        const float extrap = pos / pp.x;
        float theta = extrap;
        if (linear) {
            theta = extrap / factor;
        } else if (yarn) {
            const float interp = extrap / factor;
            theta = interp * pp.y + extrap * (1.0f - pp.y);
        } else if (llama3) {
            if (pp.w == 1.0f) {
                theta = extrap / factor;
            } else if (pp.z >= 0.0f) {
                theta = extrap * ((1.0f - pp.z) / factor + pp.z);
            }
        }
        cos_t[i] = cosf(theta) * magnitude;
        sin_t[i] = sinf(theta) * magnitude;
    }
}

/// A caller's table in the activation dtype, widened to f32 once.
template <class T>
__global__ void rope_widen_kernel(float* __restrict__ dst, const T* __restrict__ src,
                                  int64_t rows, int64_t half, int64_t stride)
{
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < rows * half;
         i += int64_t(blockDim.x) * gridDim.x) {
        const int64_t row = i / half;
        const int64_t p = i - row * half;
        dst[i] = elem_to_float(src[row * stride + p]);
    }
}

template <class T>
__global__ void rope_row_kernel(const T* __restrict__ x, T* __restrict__ y,
                                const float* __restrict__ cos_t, const float* __restrict__ sin_t,
                                const T* __restrict__ gamma, const T* __restrict__ beta,
                                int64_t rows, int64_t seq_len, int64_t head_dim,
                                int64_t rotary_dim, int64_t half, int64_t row_stride,
                                int64_t out_stride, int64_t tbl_stride, int64_t blocks,
                                int64_t outer_per_block, float eps, bool interleaved)
{
    __shared__ float red[kCudaBlockSize / 32];
    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const int64_t outer = row / seq_len;
        const int64_t pos = row - outer * seq_len;
        const T* x_row = x + outer * seq_len * row_stride + pos * row_stride;
        T* y_row = y + outer * seq_len * out_stride + pos * out_stride;
        const int64_t tbl_row =
            (blocks > 1 ? (outer / outer_per_block) * seq_len : 0) + pos;
        const float* cos_row = cos_t + tbl_row * tbl_stride;
        const float* sin_row = sin_t + tbl_row * tbl_stride;

        if (gamma != nullptr) {
            float ss = 0.0f;
            for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                const float v = elem_to_float(x_row[d]);
                ss += v * v;
            }
            // The block reduction returns the total in lane 0 only; hand it to
            // every lane through shared memory before the apply pass.
            ss = block_reduce_sum(ss, red);
            __shared__ float s_total;
            if (threadIdx.x == 0) {
                s_total = ss;
            }
            __syncthreads();
            const float inv_rms = rsqrtf(s_total / static_cast<float>(head_dim) + eps);
            for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
                float v = elem_to_float(x_row[d]) * inv_rms * elem_to_float(gamma[d]);
                if (beta != nullptr) {
                    v += elem_to_float(beta[d]);
                }
                s_store(y_row + d, v);
            }
            __syncthreads();
            // The rotation reads the normalized row back from the output, in
            // place, exactly as the CPU's fused route does.
            if (interleaved) {
                for (int64_t d = threadIdx.x * 2; d < rotary_dim; d += int64_t(blockDim.x) * 2) {
                    const float x0 = elem_to_float(y_row[d]);
                    const float x1 = elem_to_float(y_row[d + 1]);
                    const float c = cos_row[d / 2];
                    const float s = sin_row[d / 2];
                    s_store(y_row + d, x0 * c - x1 * s);
                    s_store(y_row + d + 1, x1 * c + x0 * s);
                }
            } else {
                for (int64_t d = threadIdx.x; d < half; d += blockDim.x) {
                    const float x0 = elem_to_float(y_row[d]);
                    const float x1 = elem_to_float(y_row[d + half]);
                    const float c = cos_row[d];
                    const float s = sin_row[d];
                    s_store(y_row + d, x0 * c - x1 * s);
                    s_store(y_row + d + half, x1 * c + x0 * s);
                }
            }
            continue;
        }

        if (interleaved) {
            for (int64_t d = threadIdx.x * 2; d < rotary_dim; d += int64_t(blockDim.x) * 2) {
                const float x0 = elem_to_float(x_row[d]);
                const float x1 = elem_to_float(x_row[d + 1]);
                const float c = cos_row[d / 2];
                const float s = sin_row[d / 2];
                s_store(y_row + d, x0 * c - x1 * s);
                s_store(y_row + d + 1, x1 * c + x0 * s);
            }
        } else {
            for (int64_t d = threadIdx.x; d < half; d += blockDim.x) {
                const float x0 = elem_to_float(x_row[d]);
                const float x1 = elem_to_float(x_row[d + half]);
                const float c = cos_row[d];
                const float s = sin_row[d];
                s_store(y_row + d, x0 * c - x1 * s);
                s_store(y_row + d + half, x1 * c + x0 * s);
            }
        }
        for (int64_t d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
            y_row[d] = x_row[d];
        }
    }
}


/// The plain rotation, one warp per row: the pairs are the lanes, so a warp's
/// loads and stores are one contiguous 4 B run per pair (8 B for f32) and no
/// block reduction or barrier is needed. Eight rows ride a block.
template <class T>
__global__ void rope_warp_kernel(const T* __restrict__ x, T* __restrict__ y,
                                 const float* __restrict__ cos_t,
                                 const float* __restrict__ sin_t, int64_t rows, int64_t seq_len,
                                 int64_t head_dim, int64_t rotary_dim, int64_t half,
                                 int64_t row_stride, int64_t out_stride, int64_t tbl_stride,
                                 int64_t blocks, int64_t outer_per_block, bool interleaved)
{
    const int warps = int(blockDim.x) / 32;
    const int wid = int(threadIdx.x) / 32;
    const int lane = int(threadIdx.x) % 32;
    for (int64_t row = int64_t(blockIdx.x) * warps + wid; row < rows;
         row += int64_t(gridDim.x) * warps) {
        const int64_t outer = row / seq_len;
        const int64_t pos = row - outer * seq_len;
        const T* x_row = x + outer * seq_len * row_stride + pos * row_stride;
        T* y_row = y + outer * seq_len * out_stride + pos * out_stride;
        const int64_t tbl_row = (blocks > 1 ? (outer / outer_per_block) * seq_len : 0) + pos;
        const float* cos_row = cos_t + tbl_row * tbl_stride;
        const float* sin_row = sin_t + tbl_row * tbl_stride;

        if (interleaved) {
            for (int64_t pr = lane; pr < half; pr += 32) {
                const float c = cos_row[pr];
                const float s = sin_row[pr];
                const int64_t d = 2 * pr;
                const float x0 = elem_to_float(x_row[d]);
                const float x1 = elem_to_float(x_row[d + 1]);
                s_store(y_row + d, x0 * c - x1 * s);
                s_store(y_row + d + 1, x1 * c + x0 * s);
            }
        } else {
            for (int64_t d = lane; d < half; d += 32) {
                const float c = cos_row[d];
                const float s = sin_row[d];
                const float x0 = elem_to_float(x_row[d]);
                const float x1 = elem_to_float(x_row[d + half]);
                s_store(y_row + d, x0 * c - x1 * s);
                s_store(y_row + d + half, x1 * c + x0 * s);
            }
        }
        for (int64_t d = rotary_dim + lane; d < head_dim; d += 32) {
            y_row[d] = x_row[d];
        }
    }
}

/// A stream-ordered scratch block for the tables and the small plan arrays.
struct Scratch {
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
    void release(cudaStream_t stream)
    {
        if (ptr == nullptr) {
            return;
        }
        if (async && cudaFreeAsync(ptr, stream) == cudaSuccess) {
            ptr = nullptr;
            return;
        }
        cudaGetLastError();
        cudaFree(ptr);
        ptr = nullptr;
    }
};

inline unsigned grid_for(int64_t work) noexcept
{
    return static_cast<unsigned>(std::min<int64_t>(work > 0 ? work : 1,
                                                   int64_t{1} << 20));
}

template <class T>
Status rope_entry(const RoPEAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, cudaStream_t stream)
{
    const TensorView& X = inputs[0];
    const int64_t rank = X.rank();
    if (rank < 2) {
        return Status::invalid_argument;
    }
    const int64_t seq_len = X.shape(rank - 2);
    const int64_t head_dim = X.shape(rank - 1);
    const int64_t rotary_dim = attrs.rotary_dim > 0 ? attrs.rotary_dim : head_dim;
    if (head_dim <= 0 || rotary_dim < 2 || rotary_dim % 2 != 0 || rotary_dim > head_dim ||
        seq_len <= 0) {
        return Status::invalid_argument;
    }
    const int64_t half = rotary_dim / 2;
    int64_t outer_count = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        outer_count *= X.shape(i);
    }
    const int64_t batch = rank >= 3 ? X.shape(0) : 1;
    const bool has_table = inputs.size() > 3 && !inputs[3].is_empty();
    const T* gamma = inputs.size() > 1 && !inputs[1].is_empty() ? inputs[1].ptr<T>() : nullptr;
    const T* beta = inputs.size() > 2 && !inputs[2].is_empty() ? inputs[2].ptr<T>() : nullptr;

    const int64_t row_stride = X.row_stride_elems();
    const int64_t out_stride = output.row_stride_elems();
    const int64_t rows = outer_count * seq_len;

    Scratch plan;
    Scratch tables;
    const float* cos_src = nullptr;
    const float* sin_src = nullptr;
    int64_t tbl_stride = half;
    int64_t blocks = 1;
    int64_t outer_per_block = 1;

    if (has_table) {
        const TensorView& ct = inputs[3];
        const TensorView& st = inputs[4];
        if (ct.data_type() == DataType::f32) {
            cos_src = ct.ptr<float>();
            sin_src = st.ptr<float>();
            tbl_stride = ct.row_stride_elems();
        } else {
            const int64_t bytes = seq_len * half * 8;
            if (!tables.alloc(bytes, stream)) {
                return Status::out_of_memory;
            }
            float* cbuf = static_cast<float*>(tables.ptr);
            float* sbuf = cbuf + seq_len * half;
            rope_widen_kernel<T><<<grid_for((seq_len * half + kCudaBlockSize - 1) / kCudaBlockSize),
                                   kCudaBlockSize, 0, stream>>>(
                cbuf, ct.ptr<T>(), seq_len, half, ct.row_stride_elems());
            rope_widen_kernel<T><<<grid_for((seq_len * half + kCudaBlockSize - 1) / kCudaBlockSize),
                                   kCudaBlockSize, 0, stream>>>(
                sbuf, st.ptr<T>(), seq_len, half, st.row_stride_elems());
            cos_src = cbuf;
            sin_src = sbuf;
            tbl_stride = half;
        }
    } else {
        // The per-batch first-row positions. The table kernel reads them from
        // the device, so only the Dynamic pivot needs them on the host: a
        // readback there costs a synchronising copy per call (1.2 ms on
        // Windows), and every step of a decode pays it twice a layer.
        std::vector<int64_t> starts;
        const bool has_pos = inputs.size() > 5 && !inputs[5].is_empty();
        const bool need_host_start = attrs.scaling_type == RoPEScalingType::Dynamic;
        const int64_t* dev_starts = nullptr;
        if (has_pos) {
            dev_starts = inputs[5].ptr<int64_t>();
            if (inputs[5].numel() > 1) {
                blocks = batch;
            }
            if (need_host_start) {
                const int64_t n = inputs[5].numel();
                std::vector<int64_t> host(static_cast<size_t>(n));
                CUDA_CHECK(cudaMemcpyAsync(host.data(), inputs[5].ptr<int64_t>(),
                                           size_t(n) * sizeof(int64_t), cudaMemcpyDeviceToHost,
                                           stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (n == 1) {
                    starts.push_back(host[0]);
                } else {
                    starts = host;
                    bool uniform = true;
                    for (int64_t b = 1; b < batch; ++b) {
                        uniform = uniform && starts[static_cast<size_t>(b)] == starts[0];
                    }
                    if (uniform) {
                        starts.resize(1);
                        blocks = 1;
                        dev_starts = nullptr;
                    }
                }
            }
        } else {
            starts.push_back(attrs.position_offset);
        }
        int64_t max_start = 0;
        if (need_host_start) {
            max_start = starts.empty() ? 0 : starts[0];
            for (int64_t s : starts) {
                max_start = std::max(max_start, s);
            }
        }
        outer_per_block = blocks > 1 ? outer_count / batch : outer_count;
        const auto sections = plan_rope_sections(attrs, rotary_dim, seq_len, max_start);
        const auto pairs = build_rope_pair_plans(sections, attrs, half);
        const int64_t pair_bytes = half * int64_t(sizeof(float4));
        const int64_t start_bytes =
            dev_starts == nullptr ? int64_t(starts.size()) * int64_t(sizeof(int64_t)) : 0;
        if (!plan.alloc(pair_bytes + start_bytes, stream)) {
            return Status::out_of_memory;
        }
        // The plan arrays are built on the host (the scratch is device memory),
        // then uploaded once.
        float4* dpairs = static_cast<float4*>(plan.ptr);
        int64_t* dstarts = reinterpret_cast<int64_t*>(static_cast<char*>(plan.ptr) + pair_bytes);
        std::vector<float4> host_pairs(static_cast<size_t>(half));
        for (int64_t p = 0; p < half; ++p) {
            const RopePairPlan& pp = pairs[static_cast<size_t>(p)];
            host_pairs[static_cast<size_t>(p)] =
                make_float4(pp.ratio, pp.ramp, pp.smooth, pp.long_wl);
        }
        CUDA_CHECK(cudaMemcpyAsync(dpairs, host_pairs.data(), size_t(pair_bytes),
                                   cudaMemcpyHostToDevice, stream));
        if (start_bytes > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dstarts, starts.data(), size_t(start_bytes),
                                       cudaMemcpyHostToDevice, stream));
        } else {
            dstarts = const_cast<int64_t*>(dev_starts);
        }
        const int64_t table_elems = blocks * seq_len * half;
        if (!tables.alloc(table_elems * 8, stream)) {
            plan.release(stream);
            return Status::out_of_memory;
        }
        float* cbuf = static_cast<float*>(tables.ptr);
        float* sbuf = cbuf + table_elems;
        const bool yarn = attrs.scaling_type == RoPEScalingType::YaRN;
        const float magnitude =
            yarn ? yarn_attention_scale(attrs) : 1.0f;
        rope_table_kernel<<<grid_for((table_elems + kCudaBlockSize - 1) / kCudaBlockSize),
                            kCudaBlockSize, 0, stream>>>(
            cbuf, sbuf, dpairs, dstarts, blocks * seq_len, seq_len, half,
            attrs.scaling_type == RoPEScalingType::Linear ? 1 : 0, attrs.scaling_factor,
            magnitude, yarn ? 1 : 0, attrs.scaling_type == RoPEScalingType::Llama3 ? 1 : 0);
        cos_src = cbuf;
        sin_src = sbuf;
        tbl_stride = half;
    }
    if (gamma == nullptr) {
        constexpr int kWarps = 8;
        const unsigned blocks_w = grid_for((rows + kWarps - 1) / kWarps);
        rope_warp_kernel<T><<<blocks_w, kWarps * 32, 0, stream>>>(
            X.ptr<T>(), output.ptr<T>(), cos_src, sin_src, rows, seq_len, head_dim, rotary_dim,
            half, row_stride, out_stride, tbl_stride, blocks, outer_per_block,
            attrs.interleaved);
    } else {
        rope_row_kernel<T><<<grid_for(rows), kThreads, 0, stream>>>(
            X.ptr<T>(), output.ptr<T>(), cos_src, sin_src, gamma, beta, rows, seq_len, head_dim,
            rotary_dim, half, row_stride, out_stride, tbl_stride, blocks, outer_per_block,
            attrs.norm_epsilon, attrs.interleaved);
    }

    const Status st = cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
    plan.release(stream);
    tables.release(stream);
    return st;
}

} // anonymous namespace

Status rope_cuda(const RoPEAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return rope_entry<float>(attrs, output, inputs, stream);
    case DataType::f16:
        return rope_entry<__half>(attrs, output, inputs, stream);
    case DataType::bf16:
        return rope_entry<__nv_bfloat16>(attrs, output, inputs, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
