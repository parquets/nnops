/// @file roi_align.cpp
/// @brief CPU RoiAlign — bilinear ROI pooling over NHWC feature maps.
///
/// One task per ROI (the caller's parallel_for), each walking its OH × OW bins
/// and the per-bin sampling grid. The channel axis is contiguous, so one
/// bilinear tap is four contiguous pixel loads that feed every channel: the
/// four weights are computed once per sample and applied to a channel run.
///
/// f32 and f16 accumulate in f32. A tap outside [-1, H] × [-1, W] contributes
/// zero (ONNX's rule), so a bin that sits entirely outside still averages to 0
/// and maxes to 0 rather than being skipped. A batch index outside [0, N)
/// writes zeros for its ROI: it is data, which validate() cannot see, and
/// aborting the whole call over one bad row would be worse than a zero crop.

#include "nnops/ops/roi_align.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "common/dtype_dispatch.hpp"
#include "common/memory_pool.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

template <typename T>
void roi_align_impl(const RoiAlignAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& X = inputs[0];
    const auto& rois = inputs[1];
    const bool has_batch = inputs.size() >= 3;

    const int64_t N = X.shape(0);
    const int64_t H = X.shape(1);
    const int64_t W = X.shape(2);
    const int64_t C = X.shape(3);
    const int64_t R = rois.shape(0);
    const int64_t OH = attrs.output_height;
    const int64_t OW = attrs.output_width;
    const float scale = attrs.spatial_scale;
    const bool half_pixel = attrs.coordinate_mode == RoiAlignCoordinateMode::HalfPixel;
    const bool max_mode = attrs.mode == RoiAlignMode::Max;
    const bool add_to = attrs.add_to;

    if (R == 0 || C == 0) {
        return;
    }

    const auto* x_ptr = X.ptr<T>();
    const auto* roi_ptr = rois.ptr<float>();
    auto* y_ptr = output.ptr<T>();

    const int64_t xs0 = X.stride_elems(0);
    const int64_t xs1 = X.stride_elems(1);
    const int64_t xs2 = X.stride_elems(2);
    const int64_t ys0 = output.stride_elems(0);
    const int64_t ys1 = output.stride_elems(1);
    const int64_t ys2 = output.stride_elems(2);

    const bool batch_s32 = has_batch && inputs[2].data_type() == DataType::s32;
    const auto* b32 = (has_batch && batch_s32) ? inputs[2].ptr<int32_t>() : nullptr;
    const auto* b64 = (has_batch && !batch_s32) ? inputs[2].ptr<int64_t>() : nullptr;

    const float neg_inf = -std::numeric_limits<float>::infinity();

    ctx.cpu.run(0, R, [&](int64_t r) {
        int64_t b = 0;
        if (has_batch) {
            b = batch_s32 ? static_cast<int64_t>(b32[r]) : b64[r];
        }
        T* y_roi = y_ptr + r * ys0;

        if (b < 0 || b >= N) {
            if (!add_to) {
                for (int64_t oh = 0; oh < OH; ++oh) {
                    for (int64_t ow = 0; ow < OW; ++ow) {
                        T* yp = y_roi + oh * ys1 + ow * ys2;
                        for (int64_t c = 0; c < C; ++c) {
                            s_store(&yp[c], 0.0f);
                        }
                    }
                }
            }
            return;
        }

        const T* x_batch = x_ptr + b * xs0;
        const float* roi = roi_ptr + r * 4;

        float start_w = roi[0] * scale;
        float start_h = roi[1] * scale;
        float end_w = roi[2] * scale;
        float end_h = roi[3] * scale;
        if (half_pixel) {
            start_w -= 0.5f;
            start_h -= 0.5f;
            end_w -= 0.5f;
            end_h -= 0.5f;
        }

        const float roi_w = end_w - start_w;
        const float roi_h = end_h - start_h;
        const float bin_w = roi_w / static_cast<float>(OW);
        const float bin_h = roi_h / static_cast<float>(OH);

        // sampling_ratio 0 is adaptive: one sample per input unit of the bin.
        const int64_t grid_h =
            attrs.sampling_ratio > 0
                ? attrs.sampling_ratio
                : std::max<int64_t>(1, static_cast<int64_t>(std::ceil(roi_h / OH)));
        const int64_t grid_w =
            attrs.sampling_ratio > 0
                ? attrs.sampling_ratio
                : std::max<int64_t>(1, static_cast<int64_t>(std::ceil(roi_w / OW)));
        const float inv_count =
            1.0f / static_cast<float>(grid_h * grid_w);

        PoolArray<float> acc_arr(static_cast<size_t>(C));
        float* acc = acc_arr.data();

        for (int64_t ph = 0; ph < OH; ++ph) {
            for (int64_t pw = 0; pw < OW; ++pw) {
                std::fill_n(acc, C, max_mode ? neg_inf : 0.0f);

                for (int64_t iy = 0; iy < grid_h; ++iy) {
                    const float y =
                        start_h + static_cast<float>(ph) * bin_h +
                        (static_cast<float>(iy) + 0.5f) * bin_h / static_cast<float>(grid_h);
                    for (int64_t ix = 0; ix < grid_w; ++ix) {
                        const float x =
                            start_w + static_cast<float>(pw) * bin_w +
                            (static_cast<float>(ix) + 0.5f) * bin_w /
                                static_cast<float>(grid_w);

                        // Outside the padded image the bilinear tap is zero.
                        if (y < -1.0f || y > static_cast<float>(H) || x < -1.0f ||
                            x > static_cast<float>(W)) {
                            if (max_mode) {
                                for (int64_t c = 0; c < C; ++c) {
                                    acc[c] = std::max(acc[c], 0.0f);
                                }
                            }
                            continue;
                        }

                        const float yc = std::max(y, 0.0f);
                        const float xc = std::max(x, 0.0f);
                        int64_t y_low = static_cast<int64_t>(std::floor(yc));
                        int64_t x_low = static_cast<int64_t>(std::floor(xc));
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

                        const T* p00 = x_batch + y_low * xs1 + x_low * xs2;
                        const T* p01 = x_batch + y_low * xs1 + x_high * xs2;
                        const T* p10 = x_batch + y_high * xs1 + x_low * xs2;
                        const T* p11 = x_batch + y_high * xs1 + x_high * xs2;

                        if (max_mode) {
                            for (int64_t c = 0; c < C; ++c) {
                                const float v = w00 * s_load(&p00[c]) + w01 * s_load(&p01[c]) +
                                                w10 * s_load(&p10[c]) + w11 * s_load(&p11[c]);
                                acc[c] = std::max(acc[c], v);
                            }
                        } else {
                            for (int64_t c = 0; c < C; ++c) {
                                acc[c] += w00 * s_load(&p00[c]) + w01 * s_load(&p01[c]) +
                                          w10 * s_load(&p10[c]) + w11 * s_load(&p11[c]);
                            }
                        }
                    }
                }

                T* yp = y_roi + ph * ys1 + pw * ys2;
                if (max_mode) {
                    for (int64_t c = 0; c < C; ++c) {
                        s_store_add(&yp[c], acc[c], add_to);
                    }
                } else {
                    for (int64_t c = 0; c < C; ++c) {
                        s_store_add(&yp[c], acc[c] * inv_count, add_to);
                    }
                }
            }
        }
    });
}

} // anonymous namespace

Status roi_align_cpu(const RoiAlignAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx,
                     void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "roi_align_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        roi_align_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
