/// @file norm.cu
/// @brief CUDA normalization — BatchNorm, LayerNorm, RMSNorm, GroupNorm, L2Norm.
///
/// One entry point dispatches on NormAttributes::type, as the CPU route does,
/// and every kernel reads the view's strides (see axis_walk.hpp) rather than
/// assuming a dense tail: a padded pitch and a channel-last view are the same
/// walk with different groups, so neither needs a kernel of its own.
///
/// LayerNorm / RMSNorm / L2Norm run one block per normalization row: the block
/// folds sum and sum-of-squares over the row, then applies the affine pass with
/// the row's own element offsets. The row is the dims from `axis` on, so an axis
/// short of the last one is the same kernel with a decoded offset group; a
/// pre-added residual rides the same walk, one row at a time. A row that fits in
/// shared memory is left there by the folding pass, so the affine pass re-reads
/// it from smem instead of global memory.
///
/// Every walk here is a scalar `s_load`/`s_store` unless its group flattens
/// dense and lands on the 16 B bound, in which case a 128-bit lane covers 4
/// elements of an f32 row or 8 of a 2-byte one. The tail of such a row, and a row
/// that does not qualify, keep the scalar form.
///
/// BatchNorm is one block per (sample, channel) plane — or one thread per plane
/// when the plane is a handful of elements — with the per-channel factors folded
/// once per block. GroupNorm is one block per (sample, group); the channel step
/// and the spatial walk come from the view, which covers channel-first and
/// channel-last in the same kernel.
///
/// f32, f16 and bf16 in, f32 compute. s8/u8 input stays on the CPU (its dequantize /
/// requantize kernels live there), so a quantized X answers Status::unsupported.
/// Workspace: not used.

#include "cuda_common.cuh"
#include "nnops/ops/norm.hpp"
#include "nnops/detail/axis_walk.hpp"

#include <algorithm>

namespace nnops::backend::cuda {
namespace {

using detail::AxisGroup;
using detail::axis_group_offset;

// ============================================================
// Geometry
// ============================================================

/// One normalization row: where it starts, how long it is, how its elements and
/// its output land. `rows` is the product of the dims before the axis.
struct RowWalk {
    int64_t rows = 0;
    int64_t size = 0;
    AxisGroup pre;
    AxisGroup post;
    AxisGroup out_pre;
    AxisGroup out_post;
    AxisGroup res_pre;
    AxisGroup res_post;
};

/// The per-channel norms walk (sample, channel, spatial) directly.
struct SpatWalk {
    int64_t n_step = 0;
    int64_t c_step = 0;
    int64_t out_n_step = 0;
    int64_t out_c_step = 0;
    AxisGroup sp;
    AxisGroup out_sp;
};

inline AxisGroup make_head_group(const TensorView& v, int64_t ax)
{
    AxisGroup g;
    if (ax > 0) {
        detail::axis_fill_group(g, v, 0, ax, v.stride_elems(ax - 1));
    } else {
        detail::axis_fill_group(g, v, 0, 0, 0);
    }
    return g;
}

inline RowWalk make_row_walk(const TensorView& x, const TensorView& y, int64_t ax,
                             const TensorView* res)
{
    RowWalk w;
    const int64_t rank = x.rank();
    w.rows = 1;
    for (int64_t d = 0; d < ax; ++d) { w.rows *= x.shape(d); }
    w.size = 1;
    for (int64_t d = ax; d < rank; ++d) { w.size *= x.shape(d); }

    w.pre = make_head_group(x, ax);
    w.out_pre = make_head_group(y, ax);
    detail::axis_fill_group_unit(w.post, x, ax, rank);
    detail::axis_fill_group_unit(w.out_post, y, ax, y.rank());
    if (res != nullptr) {
        w.res_pre = make_head_group(*res, ax);
        detail::axis_fill_group_unit(w.res_post, *res, ax, res->rank());
    }
    return w;
}

inline SpatWalk make_spat_walk(const TensorView& x, const TensorView& y)
{
    SpatWalk w;
    const int64_t rank = x.rank();
    w.n_step = x.stride_elems(0);
    w.c_step = (rank >= 2) ? x.stride_elems(1) : 0;
    w.out_n_step = y.stride_elems(0);
    w.out_c_step = (rank >= 2) ? y.stride_elems(1) : 0;
    detail::axis_fill_group_unit(w.sp, x, 2, rank);
    detail::axis_fill_group_unit(w.out_sp, y, 2, y.rank());
    return w;
}

// ============================================================
// Block reduction: two sums, broadcast to every thread
// ============================================================

/// Sums `a` and `b` across the block and hands the totals back to all threads
/// through `smem` (2 × the warp count floats), which is what a two-pass row
/// kernel needs — a shuffle tree alone leaves the value in the leaders only.
__device__ inline void block_sums2(float& a, float& b, float* smem) noexcept
{
    constexpr int kWarpCap = kCudaBlockSize / kCudaWarpSize;
    const int lane = int(threadIdx.x) % kCudaWarpSize;
    const int wid = int(threadIdx.x) / kCudaWarpSize;
    const int warps = int(ceil_div(int(blockDim.x), kCudaWarpSize));

    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
    if (lane == 0) {
        smem[wid] = a;
        smem[kWarpCap + wid] = b;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float ta = 0.0f;
        float tb = 0.0f;
        for (int i = 0; i < warps; ++i) {
            ta += smem[i];
            tb += smem[kWarpCap + i];
        }
        smem[0] = ta;
        smem[kWarpCap] = tb;
    }
    __syncthreads();

    a = smem[0];
    b = smem[kWarpCap];
}

// ============================================================
// Row norms — LayerNorm / RMSNorm / L2Norm
// ============================================================

/// HasMean: subtract the row mean (LayerNorm). HasScale: scale, and RMS's
/// epsilon placement. Without either it is L2Norm, whose epsilon sits under the
/// root of the plain sum.
///
/// The row is read once when it fits in shared memory (row_resident): pass 1
/// leaves it there, so the affine pass reads smem instead of the input and the
/// residual a second time. A row past the budget is read twice, as before.
///
/// Both passes move a 128-bit lane per step whenever every side of the row
/// flattens dense and lands on the 16 B bound — the tail and a row that does not
/// (a stranded axis, a pitched or channel-last post group) keep the scalar walk.
template <class T, bool HasMean, bool HasScale, bool HasBias, bool HasRes>
__global__ void norm_row_kernel(const T* __restrict__ x,
                                const T* __restrict__ scale,
                                const T* __restrict__ bias,
                                const T* __restrict__ res,
                                T* __restrict__ y, RowWalk w, float eps,
                                bool scale_scalar, bool add_to,
                                bool row_resident)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4;

    extern __shared__ float shared_buf[];
    float* row  = row_resident ? shared_buf : nullptr;
    float* smem = shared_buf + (row_resident ? w.size : 0);

    // A dense post group puts element i at base + i; anything else is decoded.
    const bool x_flat = w.post.n == 0 && w.post.flat_step == 1;
    const bool y_flat = w.out_post.n == 0 && w.out_post.flat_step == 1;
    const bool r_flat = !HasRes || (w.res_post.n == 0 && w.res_post.flat_step == 1);
    // The factors are one per element of the row, so the same lane covers them
    // off their own base unless that base is off the bound (a scalar scale
    // broadcasts instead).
    const bool s_lane = HasScale && !scale_scalar && aligned16(scale);
    const bool b_lane = HasBias && !scale_scalar && aligned16(bias);

    for (int64_t ri = blockIdx.x; ri < w.rows; ri += gridDim.x) {
        const int64_t xb = axis_group_offset(ri, w.pre);
        const int64_t rb = HasRes ? axis_group_offset(ri, w.res_pre) : 0;
        const int64_t yb = axis_group_offset(ri, w.out_pre);

        const T* x_row = x + xb;
        const T* r_row = HasRes ? res + rb : nullptr;
        T*       y_row = y + yb;

        const int64_t vec_end =
            (x_flat && y_flat && r_flat && aligned16(x_row, y_row) &&
             (!HasRes || aligned16(r_row)))
                ? (w.size / VEC) * VEC
                : 0;

        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int64_t i = int64_t(threadIdx.x) * VEC; i < vec_end;
             i += int64_t(blockDim.x) * VEC) {
            float v[VEC];
            load_channels<T, VEC>(&x_row[i], v);
            if (HasRes) {
                float r[VEC];
                load_channels<T, VEC>(&r_row[i], r);
                #pragma unroll
                for (int k = 0; k < VEC; ++k) { v[k] += r[k]; }
            }
            #pragma unroll
            for (int k = 0; k < VEC; ++k) { sum += v[k]; sum_sq += v[k] * v[k]; }
            if (row != nullptr) {
                #pragma unroll
                for (int k = 0; k < VEC; ++k) { row[i + k] = v[k]; }
            }
        }
        for (int64_t i = vec_end + threadIdx.x; i < w.size; i += blockDim.x) {
            float v = s_load(&x[xb + axis_group_offset(i, w.post)]);
            if (HasRes) {
                v += s_load(&res[rb + axis_group_offset(i, w.res_post)]);
            }
            sum += v;
            sum_sq += v * v;
            if (row != nullptr) { row[i] = v; }
        }
        block_sums2(sum, sum_sq, smem);

        const float inv_n = 1.0f / static_cast<float>(w.size);
        float mean = 0.0f;
        float inv_std;
        if (HasMean) {
            mean = sum * inv_n;
            float var = sum_sq * inv_n - mean * mean;
            if (var < 0.0f) { var = 0.0f; }
            inv_std = rsqrtf(var + eps);
        } else if (HasScale) {
            inv_std = rsqrtf(sum_sq * inv_n + eps);
        } else {
            inv_std = rsqrtf(sum_sq + eps);
        }

        for (int64_t i = int64_t(threadIdx.x) * VEC; i < vec_end;
             i += int64_t(blockDim.x) * VEC) {
            float v[VEC];
            if (row != nullptr) {
                #pragma unroll
                for (int k = 0; k < VEC; ++k) { v[k] = row[i + k]; }
            } else {
                load_channels<T, VEC>(&x_row[i], v);
                if (HasRes) {
                    float r[VEC];
                    load_channels<T, VEC>(&r_row[i], r);
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k) { v[k] += r[k]; }
                }
            }
            #pragma unroll
            for (int k = 0; k < VEC; ++k) { v[k] = (v[k] - mean) * inv_std; }

            if (HasScale) {
                if (scale_scalar) {
                    const float s = s_load(scale);
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k) { v[k] *= s; }
                } else if (s_lane) {
                    float s[VEC];
                    load_channels<T, VEC>(&scale[i], s);
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k) { v[k] *= s[k]; }
                } else {
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k) { v[k] *= s_load(&scale[i + k]); }
                }
            }
            if (HasBias) {
                if (scale_scalar) {
                    const float b = s_load(bias);
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k) { v[k] += b; }
                } else if (b_lane) {
                    float b[VEC];
                    load_channels<T, VEC>(&bias[i], b);
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k) { v[k] += b[k]; }
                } else {
                    #pragma unroll
                    for (int k = 0; k < VEC; ++k) { v[k] += s_load(&bias[i + k]); }
                }
            }

            if (add_to) {
                float o[VEC];
                load_channels<T, VEC>(&y_row[i], o);
                #pragma unroll
                for (int k = 0; k < VEC; ++k) { v[k] += o[k]; }
            }
            store_channels<T, VEC>(&y_row[i], v);
        }

        for (int64_t i = vec_end + threadIdx.x; i < w.size; i += blockDim.x) {
            float v = (row != nullptr)
                ? row[i]
                : s_load(&x[xb + axis_group_offset(i, w.post)]);
            if (row == nullptr && HasRes) {
                v += s_load(&res[rb + axis_group_offset(i, w.res_post)]);
            }
            float val = (v - mean) * inv_std;
            const int64_t si = scale_scalar ? 0 : i;
            if (HasScale) { val *= s_load(&scale[si]); }
            if (HasBias) { val += s_load(&bias[si]); }

            const int64_t oe = yb + axis_group_offset(i, w.out_post);
            if (add_to) {
                s_store(&y[oe], s_load(&y[oe]) + val);
            } else {
                s_store(&y[oe], val);
            }
        }
    }
}

// ============================================================
// BatchNorm — per channel, pre-computed statistics
// ============================================================

/// A block per (sample, channel) plane. The factors are folded once per block;
/// rank above two means the plane has an interior to walk, and a dense one of
/// those moves a 128-bit lane per step.
template <class T>
__global__ void batch_norm_plane_block_kernel(const T* __restrict__ x,
                                              const T* __restrict__ scale,
                                              const T* __restrict__ bias,
                                              const T* __restrict__ mean,
                                              const T* __restrict__ var,
                                              T* __restrict__ y, int64_t rows,
                                              int64_t channels, int64_t plane,
                                              SpatWalk w, float eps, bool add_to)
{
    constexpr int VEC = sizeof(T) == 2 ? 8 : 4;
    const bool flat = w.sp.n == 0 && w.sp.flat_step == 1 &&
                      w.out_sp.n == 0 && w.out_sp.flat_step == 1;

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const int64_t n = row / channels;
        const int64_t c = row - n * channels;
        const float ns = s_load(&scale[c]) * rsqrtf(s_load(&var[c]) + eps);
        const float nb = s_load(&bias[c]) - s_load(&mean[c]) * ns;
        const int64_t xb = n * w.n_step + c * w.c_step;
        const int64_t yb = n * w.out_n_step + c * w.out_c_step;

        const T* x_plane = x + xb;
        T*       y_plane = y + yb;
        const int64_t vec_end =
            (flat && aligned16(x_plane, y_plane)) ? (plane / VEC) * VEC : 0;

        for (int64_t p = int64_t(threadIdx.x) * VEC; p < vec_end;
             p += int64_t(blockDim.x) * VEC) {
            float v[VEC];
            load_channels<T, VEC>(&x_plane[p], v);
            #pragma unroll
            for (int k = 0; k < VEC; ++k) { v[k] = v[k] * ns + nb; }
            if (add_to) {
                float o[VEC];
                load_channels<T, VEC>(&y_plane[p], o);
                #pragma unroll
                for (int k = 0; k < VEC; ++k) { v[k] += o[k]; }
            }
            store_channels<T, VEC>(&y_plane[p], v);
        }

        for (int64_t p = vec_end + threadIdx.x; p < plane; p += blockDim.x) {
            const float v = s_load(&x[xb + axis_group_offset(p, w.sp)]) * ns + nb;
            const int64_t oe = yb + axis_group_offset(p, w.out_sp);
            s_store(&y[oe], add_to ? s_load(&y[oe]) + v : v);
        }
    }
}

/// A thread per plane, for the shapes whose plane is one element or a few — the
/// block-per-plane kernel would spend a block on them.
template <class T>
__global__ void batch_norm_plane_thread_kernel(const T* __restrict__ x,
                                               const T* __restrict__ scale,
                                               const T* __restrict__ bias,
                                               const T* __restrict__ mean,
                                               const T* __restrict__ var,
                                               T* __restrict__ y, int64_t rows,
                                               int64_t channels, int64_t plane,
                                               SpatWalk w, float eps, bool add_to)
{
    const int64_t step = int64_t(gridDim.x) * blockDim.x;
    for (int64_t row = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; row < rows;
         row += step) {
        const int64_t n = row / channels;
        const int64_t c = row - n * channels;
        const float ns = s_load(&scale[c]) * rsqrtf(s_load(&var[c]) + eps);
        const float nb = s_load(&bias[c]) - s_load(&mean[c]) * ns;
        const int64_t xb = n * w.n_step + c * w.c_step;
        const int64_t yb = n * w.out_n_step + c * w.out_c_step;

        for (int64_t p = 0; p < plane; ++p) {
            const float v = s_load(&x[xb + axis_group_offset(p, w.sp)]) * ns + nb;
            const int64_t oe = yb + axis_group_offset(p, w.out_sp);
            s_store(&y[oe], add_to ? s_load(&y[oe]) + v : v);
        }
    }
}

/// Non-spatial: scale / bias / mean / var are per element, so the pass is a flat
/// elementwise map (the CPU route walks the same linear range).
template <class T>
__global__ void batch_norm_element_kernel(const T* __restrict__ x,
                                          const T* __restrict__ scale,
                                          const T* __restrict__ bias,
                                          const T* __restrict__ mean,
                                          const T* __restrict__ var,
                                          T* __restrict__ y, int64_t n, float eps,
                                          bool add_to)
{
    const int64_t step = int64_t(gridDim.x) * blockDim.x;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += step) {
        const float ns = s_load(&scale[i]) * rsqrtf(s_load(&var[i]) + eps);
        const float nb = s_load(&bias[i]) - s_load(&mean[i]) * ns;
        const float v = s_load(&x[i]) * ns + nb;
        s_store(&y[i], add_to ? s_load(&y[i]) + v : v);
    }
}

// ============================================================
// GroupNorm — one block per (sample, group)
// ============================================================

template <class T, bool HasBias>
__global__ void group_norm_kernel(const T* __restrict__ x,
                                  const T* __restrict__ scale,
                                  const T* __restrict__ bias,
                                  T* __restrict__ y, int64_t ng, int64_t groups,
                                  int64_t cpg, int64_t spatial, SpatWalk w,
                                  float eps, bool add_to)
{
    __shared__ float smem[2 * (kCudaBlockSize / kCudaWarpSize)];

    const int64_t per_group = cpg * spatial;
    const float inv_n = 1.0f / static_cast<float>(per_group);

    for (int64_t t = blockIdx.x; t < ng; t += gridDim.x) {
        const int64_t n = t / groups;
        const int64_t gi = t - n * groups;
        const int64_t c_start = gi * cpg;
        const int64_t xb = n * w.n_step + c_start * w.c_step;
        const int64_t yb = n * w.out_n_step + c_start * w.out_c_step;

        // A group is channels_per_group channel planes: the channel step and the
        // spatial walk keep both layouts on the same two offsets.
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int64_t e = threadIdx.x; e < per_group; e += blockDim.x) {
            const int64_t k = e / spatial;
            const int64_t p = e - k * spatial;
            const float v = s_load(&x[xb + k * w.c_step + axis_group_offset(p, w.sp)]);
            sum += v;
            sum_sq += v * v;
        }
        block_sums2(sum, sum_sq, smem);

        const float mean = sum * inv_n;
        float var = sum_sq * inv_n - mean * mean;
        if (var < 0.0f) { var = 0.0f; }
        const float inv_std = rsqrtf(var + eps);

        for (int64_t e = threadIdx.x; e < per_group; e += blockDim.x) {
            const int64_t k = e / spatial;
            const int64_t p = e - k * spatial;
            const int64_t c = c_start + k;
            const float scale_c = s_load(&scale[c]);
            const float bias_c = HasBias ? s_load(&bias[c]) : 0.0f;

            const int64_t xe = xb + k * w.c_step + axis_group_offset(p, w.sp);
            const float v = (s_load(&x[xe]) - mean) * inv_std * scale_c + bias_c;
            const int64_t oe = yb + k * w.out_c_step + axis_group_offset(p, w.out_sp);
            s_store(&y[oe], add_to ? s_load(&y[oe]) + v : v);
        }
    }
}

// ============================================================
// Launch
// ============================================================

inline int block_for(int64_t size)
{
    int b = 32;
    while (b < kCudaBlockSize && b < size) { b <<= 1; }
    return b;
}

template <class T, bool HasMean, bool HasScale, bool HasBias, bool HasRes>
void launch_rows(const T* x, const T* scale, const T* bias, const T* res, T* y,
                 const RowWalk& w, float eps, bool scale_scalar, bool add_to,
                 cudaStream_t stream)
{
    const unsigned grid = static_cast<unsigned>(
        std::min<int64_t>(std::max<int64_t>(w.rows, 1), 65535));
    const int block_size = block_for(w.size);

    // The row is held in shared memory when it fits beside the reduction
    // scratch, which trades a second global read for occupancy.
    const int64_t row_bytes = w.size * static_cast<int64_t>(sizeof(float));
    const bool row_resident = row_resident_worth(
        w.size, static_cast<int64_t>(sizeof(T)),
        block_size * static_cast<int64_t>(sizeof(float)));
    const int shared_bytes = static_cast<int>(
        block_size * sizeof(float) + (row_resident ? row_bytes : 0));

    norm_row_kernel<T, HasMean, HasScale, HasBias, HasRes>
        <<<grid, block_size, shared_bytes, stream>>>(
            x, scale, bias, res, y, w, eps, scale_scalar, add_to, row_resident);
}

template <class T, bool HasMean, bool HasScale>
void launch_rows_bias(const T* x, const T* scale, const T* bias, const T* res,
                      T* y, const RowWalk& w, float eps, bool scale_scalar,
                      bool add_to, bool has_bias, bool has_res,
                      cudaStream_t stream)
{
    if (has_bias) {
        if (has_res) {
            launch_rows<T, HasMean, HasScale, true, true>(
                x, scale, bias, res, y, w, eps, scale_scalar, add_to, stream);
        } else {
            launch_rows<T, HasMean, HasScale, true, false>(
                x, scale, bias, nullptr, y, w, eps, scale_scalar, add_to, stream);
        }
    } else {
        if (has_res) {
            launch_rows<T, HasMean, HasScale, false, true>(
                x, scale, nullptr, res, y, w, eps, scale_scalar, add_to, stream);
        } else {
            launch_rows<T, HasMean, HasScale, false, false>(
                x, scale, nullptr, nullptr, y, w, eps, scale_scalar, add_to, stream);
        }
    }
}

template <class T, bool HasMean, bool HasScale>
Status run_row_norm(const NormAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, int64_t scale_slot,
                  int64_t bias_slot, int64_t res_slot, cudaStream_t stream)
{
    const auto& x = inputs[0];
    auto& y = output;

    const TensorView* scale = (scale_slot >= 0) ? &inputs[scale_slot] : nullptr;
    // A slot past the end of the call is absent, not empty: LayerNorm and
    // RMSNorm drop the bias entirely unless a residual holds the slot.
    const auto present = [&](int64_t slot) {
        return slot >= 0 && slot < static_cast<int64_t>(inputs.size()) &&
               !inputs[slot].is_empty();
    };
    const bool has_bias = present(bias_slot);
    const bool has_res = present(res_slot);
    const bool scale_scalar = scale != nullptr && scale->numel() == 1;

    const int64_t ax = attrs.axis < 0 ? attrs.axis + x.rank() : attrs.axis;
    const RowWalk w = make_row_walk(x, y, ax, has_res ? &inputs[res_slot] : nullptr);
    if (w.rows == 0 || w.size == 0) {
        return Status::ok;
    }

    launch_rows_bias<T, HasMean, HasScale>(
        x.ptr<T>(), scale ? scale->ptr<T>() : nullptr,
        has_bias ? inputs[bias_slot].ptr<T>() : nullptr,
        has_res ? inputs[res_slot].ptr<T>() : nullptr, y.ptr<T>(), w,
        attrs.epsilon, scale_scalar, attrs.add_to, has_bias, has_res, stream);
    return Status::ok;
}

template <class T>
Status batch_norm_launch(const NormAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, cudaStream_t stream)
{
    const auto& x = inputs[0];
    auto& y = output;
    const int64_t rank = x.rank();
    if (rank == 0) { return Status::invalid_argument; }

    const T* s_ptr = inputs[1].ptr<T>();
    const T* b_ptr = inputs[2].ptr<T>();
    const T* m_ptr = inputs[3].ptr<T>();
    const T* v_ptr = inputs[4].ptr<T>();

    if (!attrs.spatial) {
        const int64_t total = x.numel();
        if (total == 0) { return Status::ok; }
        const unsigned grid = static_cast<unsigned>(
            std::min<int64_t>(ceil_div(total, int64_t(kCudaBlockSize)), 65535));
        batch_norm_element_kernel<T><<<grid, kCudaBlockSize, 0, stream>>>(
            x.ptr<T>(), s_ptr, b_ptr, m_ptr, v_ptr, y.ptr<T>(), total,
            attrs.epsilon, attrs.add_to);
        return Status::ok;
    }

    const int64_t N = x.shape(0);
    const int64_t C = (rank >= 2) ? x.shape(1) : 1;
    int64_t plane = 1;
    for (int64_t d = 2; d < rank; ++d) { plane *= x.shape(d); }
    // Rank one has no channel axis of its own, and rank two's channel is the
    // last dim: both leave one element per (sample, channel) plane.
    const int64_t rows = N * C;
    if (rows == 0 || plane == 0) { return Status::ok; }

    const SpatWalk w = make_spat_walk(x, y);
    if (plane <= 32) {
        const unsigned grid = static_cast<unsigned>(
            std::min<int64_t>(ceil_div(rows, int64_t(kCudaBlockSize)), 65535));
        batch_norm_plane_thread_kernel<T><<<grid, kCudaBlockSize, 0, stream>>>(
            x.ptr<T>(), s_ptr, b_ptr, m_ptr, v_ptr, y.ptr<T>(), rows, C, plane, w,
            attrs.epsilon, attrs.add_to);
    } else {
        const unsigned grid = static_cast<unsigned>(
            std::min<int64_t>(rows, 65535));
        batch_norm_plane_block_kernel<T><<<grid, kCudaBlockSize, 0, stream>>>(
            x.ptr<T>(), s_ptr, b_ptr, m_ptr, v_ptr, y.ptr<T>(), rows, C, plane, w,
            attrs.epsilon, attrs.add_to);
    }
    return Status::ok;
}

template <class T>
Status group_norm_launch(const NormAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, cudaStream_t stream)
{
    const auto& x = inputs[0];
    auto& y = output;
    const int64_t rank = x.rank();
    if (rank < 2) { return Status::invalid_argument; }

    const int64_t N = x.shape(0);
    const int64_t C = x.shape(1);
    const int64_t G = attrs.num_groups > 0 ? attrs.num_groups : 1;
    if (G <= 0 || C % G != 0) { return Status::invalid_argument; }
    const int64_t cpg = C / G;

    int64_t spatial = 1;
    for (int64_t d = 2; d < rank; ++d) { spatial *= x.shape(d); }
    if (N == 0 || spatial == 0 || cpg == 0) { return Status::ok; }

    const bool has_bias = inputs.size() >= 3 && !inputs[2].is_empty();
    const SpatWalk w = make_spat_walk(x, y);
    const unsigned grid = static_cast<unsigned>(
        std::min<int64_t>(N * G, 65535));

    if (has_bias) {
        group_norm_kernel<T, true><<<grid, kCudaBlockSize, 0, stream>>>(
            x.ptr<T>(), inputs[1].ptr<T>(), inputs[2].ptr<T>(), y.ptr<T>(), N * G,
            G, cpg, spatial, w, attrs.epsilon, attrs.add_to);
    } else {
        group_norm_kernel<T, false><<<grid, kCudaBlockSize, 0, stream>>>(
            x.ptr<T>(), inputs[1].ptr<T>(), nullptr, y.ptr<T>(), N * G, G, cpg,
            spatial, w, attrs.epsilon, attrs.add_to);
    }
    return Status::ok;
}

template <class T>
Status norm_entry(const NormAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, cudaStream_t stream)
{
    switch (attrs.type) {
    case NormType::BatchNorm:
        return batch_norm_launch<T>(attrs, output, inputs, stream);
    case NormType::LayerNorm:
        // [X, scale, bias?, residual?]
        return run_row_norm<T, true, true>(attrs, output, inputs, 1, 2,
                                           attrs.add_residual ? 3 : -1, stream);
    case NormType::RMSNorm:
        // [X, scale, residual?]
        return run_row_norm<T, false, true>(attrs, output, inputs, 1, -1,
                                            attrs.add_residual ? 2 : -1, stream);
    case NormType::GroupNorm:
        return group_norm_launch<T>(attrs, output, inputs, stream);
    case NormType::L2Norm:
        // [X, residual?]
        return run_row_norm<T, false, false>(attrs, output, inputs, -1, -1,
                                             attrs.add_residual ? 1 : -1, stream);
    }
    return Status::invalid_argument;
}

}  // anonymous namespace

// ============================================================
// Entry point
// ============================================================

Status norm_cuda(const NormAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx,
                 void* /*workspace*/)
{
    const auto& x = inputs[0];
    if (is_quantized_dtype(x.data_type())) { return Status::unsupported; }

    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (x.data_type()) {
    case DataType::f32:
        return norm_entry<float>(attrs, output, inputs, stream);
    case DataType::f16:
        return norm_entry<__half>(attrs, output, inputs, stream);
    case DataType::bf16:
        return norm_entry<__nv_bfloat16>(attrs, output, inputs, stream);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
