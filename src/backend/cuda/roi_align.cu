/// @file roi_align.cu
/// @brief CUDA RoiAlign — bilinear ROI pooling over NHWC feature maps.
///
/// One block per (roi, output bin) and the lanes span C, which is exactly the
/// CPU route's split (a task per ROI, a SIMD lane across C). Each lane owns a
/// small channel group: the four bilinear taps of a sample are vector loads at
/// that group, so the weights are computed once per sample per group rather
/// than once per channel, and the store is one vector.
///
/// The tap rules are the CPU's: a sample outside `[-1, H] × [-1, W]` contributes
/// zero (and under Max contributes a zero to the running maximum), the four
/// corners clamp at the far border, HalfPixel pulls the ROI corners in by 0.5,
/// and OutputHalfPixel forces a degenerate ROI to 1x1. A batch index outside
/// [0, N) writes zeros for the ROI. f32 and f16 accumulate in f32; add_to adds.

#include "cuda_common.cuh"
#include "nnops/ops/roi_align.hpp"

#include <limits>

namespace nnops::backend::cuda {

namespace {

constexpr int kThreads = 256;

__device__ inline float ld(const float* p) { return *p; }
__device__ inline float ld(const __half* p) { return __half2float(*p); }

template <class T> __device__ inline void st_add(T* p, float v, bool add_to)
{
    s_store(p, add_to ? elem_to_float(*p) + v : v);
}

template <class T>
__global__ void roi_align_kernel(const T* __restrict__ x, const float* __restrict__ rois,
                                 const void* __restrict__ batch, bool batch_s32,
                                 T* __restrict__ y, int64_t bins, int64_t N, int64_t H, int64_t W,
                                 int64_t C, int64_t OH, int64_t OW, int64_t xs0, int64_t xs1,
                                 int64_t xs2, int64_t ys0, int64_t ys1, int64_t ys2, float scale,
                                 bool half_pixel, bool max_mode, int64_t sampling_ratio,
                                 bool add_to)
{
    constexpr int VEC = std::is_same_v<T, float> ? 4 : 8;
    const float neg_inf = -INFINITY;

    for (int64_t bin = blockIdx.x; bin < bins; bin += gridDim.x) {
        int64_t r = bin;
        const int64_t pw = r % OW;
        r /= OW;
        const int64_t ph = r % OH;
        const int64_t roi = r / OH;

        T* y_bin = y + roi * ys0 + ph * ys1 + pw * ys2;

        int64_t b = 0;
        if (batch != nullptr) {
            b = batch_s32 ? static_cast<int64_t>(static_cast<const int32_t*>(batch)[roi])
                          : static_cast<const int64_t*>(batch)[roi];
        }
        if (b < 0 || b >= N) {
            for (int64_t c = threadIdx.x; c < C; c += blockDim.x) {
                st_add(y_bin + c, 0.0f, add_to);
            }
            continue;
        }

        const T* xb = x + b * xs0;
        const float* roi_v = rois + roi * 4;
        float start_w = roi_v[0] * scale;
        float start_h = roi_v[1] * scale;
        float end_w = roi_v[2] * scale;
        float end_h = roi_v[3] * scale;
        if (half_pixel) {
            start_w -= 0.5f;
            start_h -= 0.5f;
            end_w -= 0.5f;
            end_h -= 0.5f;
        }
        float roi_w = end_w - start_w;
        float roi_h = end_h - start_h;
        if (!half_pixel) {
            roi_w = fmaxf(roi_w, 1.0f);
            roi_h = fmaxf(roi_h, 1.0f);
        }
        const float bin_w = roi_w / static_cast<float>(OW);
        const float bin_h = roi_h / static_cast<float>(OH);
        // sampling_ratio 0 is adaptive: one sample per input unit of the bin,
        // recomputed per ROI from its own extents.
        const int64_t grid_h = sampling_ratio > 0 ? sampling_ratio
                                                  : (int64_t)ceilf(roi_h / (float)OH);
        const int64_t grid_w = sampling_ratio > 0 ? sampling_ratio
                                                  : (int64_t)ceilf(roi_w / (float)OW);
        const int64_t gh = grid_h > 0 ? grid_h : 1;
        const int64_t gw = grid_w > 0 ? grid_w : 1;
        const float inv_count = 1.0f / static_cast<float>(gh * gw);

        for (int64_t c0 = int64_t(threadIdx.x) * VEC; c0 < C;
             c0 += int64_t(blockDim.x) * VEC) {
            const bool vec = c0 + VEC <= C &&
                             ((reinterpret_cast<uintptr_t>(y_bin + c0) & 15u) == 0);
            const int64_t lim = vec ? VEC : (C - c0 < VEC ? C - c0 : VEC);
            float acc[VEC];
#pragma unroll
            for (int k = 0; k < VEC; ++k) {
                acc[k] = max_mode ? neg_inf : 0.0f;
            }

            for (int64_t iy = 0; iy < gh; ++iy) {
                const float yy = start_h + static_cast<float>(ph) * bin_h +
                                 (static_cast<float>(iy) + 0.5f) * bin_h /
                                     static_cast<float>(gh);
                for (int64_t ix = 0; ix < gw; ++ix) {
                    const float xx = start_w + static_cast<float>(pw) * bin_w +
                                     (static_cast<float>(ix) + 0.5f) * bin_w /
                                         static_cast<float>(gw);
                    if (!(yy >= -1.0f && yy <= static_cast<float>(H) && xx >= -1.0f &&
                          xx <= static_cast<float>(W))) {
                        if (max_mode) {
#pragma unroll
                            for (int k = 0; k < VEC; ++k) {
                                acc[k] = fmaxf(acc[k], 0.0f);
                            }
                        }
                        continue;
                    }

                    const float yc = fmaxf(yy, 0.0f);
                    const float xc = fmaxf(xx, 0.0f);
                    int64_t y_low = static_cast<int64_t>(floorf(yc));
                    int64_t x_low = static_cast<int64_t>(floorf(xc));
                    int64_t y_high = y_low + 1;
                    int64_t x_high = x_low + 1;
                    float dy = yc - static_cast<float>(y_low);
                    float dx = xc - static_cast<float>(x_low);
                    if (y_low >= H - 1) {
                        y_high = y_low = H - 1;
                        dy = 0.0f;
                    }
                    if (x_low >= W - 1) {
                        x_high = x_low = W - 1;
                        dx = 0.0f;
                    }
                    const float w00 = (1.0f - dy) * (1.0f - dx);
                    const float w01 = (1.0f - dy) * dx;
                    const float w10 = dy * (1.0f - dx);
                    const float w11 = dy * dx;

                    const T* p00 = xb + y_low * xs1 + x_low * xs2 + c0;
                    const T* p01 = xb + y_low * xs1 + x_high * xs2 + c0;
                    const T* p10 = xb + y_high * xs1 + x_low * xs2 + c0;
                    const T* p11 = xb + y_high * xs1 + x_high * xs2 + c0;

                    float v[VEC];
                    if (vec) {
                        if constexpr (std::is_same_v<T, float>) {
                            const float4 a = v4_load(p00);
                            const float4 bq = v4_load(p01);
                            const float4 cq = v4_load(p10);
                            const float4 d = v4_load(p11);
                            v[0] = w00 * a.x + w01 * bq.x + w10 * cq.x + w11 * d.x;
                            v[1] = w00 * a.y + w01 * bq.y + w10 * cq.y + w11 * d.y;
                            v[2] = w00 * a.z + w01 * bq.z + w10 * cq.z + w11 * d.z;
                            v[3] = w00 * a.w + w01 * bq.w + w10 * cq.w + w11 * d.w;
                        } else {
                            const pack8<T> a = v4_load(p00);
                            const pack8<T> bq = v4_load(p01);
                            const pack8<T> cq = v4_load(p10);
                            const pack8<T> d = v4_load(p11);
#pragma unroll
                            for (int k = 0; k < 8; ++k) {
                                v[k] = w00 * elem_to_float(a.data[k]) +
                                       w01 * elem_to_float(bq.data[k]) +
                                       w10 * elem_to_float(cq.data[k]) +
                                       w11 * elem_to_float(d.data[k]);
                            }
                        }
                    } else {
                        for (int64_t k = 0; k < lim; ++k) {
                            v[k] = w00 * ld(p00 + k) + w01 * ld(p01 + k) + w10 * ld(p10 + k) +
                                   w11 * ld(p11 + k);
                        }
                    }
#pragma unroll
                    for (int k = 0; k < VEC; ++k) {
                        if (max_mode) {
                            acc[k] = fmaxf(acc[k], v[k]);
                        } else {
                            acc[k] += v[k];
                        }
                    }
                }
            }

            if (vec) {
                if constexpr (std::is_same_v<T, float>) {
                    float o[4];
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        o[k] = max_mode ? acc[k] : acc[k] * inv_count;
                        if (add_to) {
                            o[k] += y_bin[c0 + k];
                        }
                    }
                    v4_store(y_bin + c0, make_float4(o[0], o[1], o[2], o[3]));
                } else {
                    float o[8];
#pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        o[k] = max_mode ? acc[k] : acc[k] * inv_count;
                        if (add_to) {
                            o[k] += elem_to_float(y_bin[c0 + k]);
                        }
                    }
                    v4_store(y_bin + c0,
                             pack8_set<T>(o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]));
                }
            } else {
                for (int64_t k = 0; k < lim; ++k) {
                    st_add(y_bin + c0 + k, max_mode ? acc[k] : acc[k] * inv_count, add_to);
                }
            }
        }
    }
}

template <class T>
Status roi_align_entry(const RoiAlignAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, cudaStream_t stream)
{
    const TensorView& X = inputs[0];
    const TensorView& rois = inputs[1];
    if (X.rank() != 4) {
        return Status::invalid_argument;
    }
    const bool has_batch = inputs.size() >= 3 && !inputs[2].is_empty();
    const int64_t N = X.shape(0), H = X.shape(1), W = X.shape(2), C = X.shape(3);
    const int64_t R = rois.shape(0);
    const int64_t OH = attrs.output_height, OW = attrs.output_width;
    if (R <= 0 || C <= 0 || OH <= 0 || OW <= 0) {
        return Status::ok;
    }

    const float scale = attrs.spatial_scale;
    const bool half_pixel = attrs.coordinate_mode == RoiAlignCoordinateMode::HalfPixel;
    const bool max_mode = attrs.mode == RoiAlignMode::Max;

    const int64_t bins = R * OH * OW;
    const unsigned blocks = static_cast<unsigned>(std::min<int64_t>(bins, int64_t{1} << 20));
    const bool batch_s32 = has_batch && inputs[2].data_type() == DataType::s32;
    const void* batch = has_batch ? inputs[2].ptr<void>() : nullptr;

    roi_align_kernel<T><<<blocks, kThreads, 0, stream>>>(
        X.ptr<T>(), rois.ptr<float>(), batch, batch_s32, output.ptr<T>(), bins, N, H, W, C, OH,
        OW, X.stride_elems(0), X.stride_elems(1), X.stride_elems(2), output.stride_elems(0),
        output.stride_elems(1), output.stride_elems(2), scale, half_pixel, max_mode,
        attrs.sampling_ratio, attrs.add_to);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // anonymous namespace

Status roi_align_cuda(const RoiAlignAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return roi_align_entry<float>(attrs, output, inputs, stream);
    case DataType::f16:
        return roi_align_entry<__half>(attrs, output, inputs, stream);
    default:
        return Status::unsupported;
    }
}

} // namespace nnops::backend::cuda
