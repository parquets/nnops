/// @file resize.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial resize (NCHW/NCDHW layout).
///
/// For the initial implementation:
///   - Nearest mode: scalar per-element (already fast, contiguous access)
///   - Linear mode: scalar per-element with s_load/s_store
///   - N*C parallel dispatch via parallel_for
///   - f32 and f16 support

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// Clamp a source index to [0, src_size - 1].
inline int64_t clamp_idx(int64_t idx, int64_t src_size) {
    return std::max<int64_t>(0, std::min(idx, src_size - 1));
}

/// Compute source coordinate based on the coordinate transform mode.
inline float compute_src_coord(int64_t dst_idx, int64_t src_size, int64_t dst_size,
                               CoordinateTransformMode mode) {
    if (dst_size == src_size) {
        return static_cast<float>(dst_idx);
    }
    switch (mode) {
    case CoordinateTransformMode::HalfPixel: {
        float scale = static_cast<float>(src_size) / static_cast<float>(dst_size);
        return (static_cast<float>(dst_idx) + 0.5f) * scale - 0.5f;
    }
    case CoordinateTransformMode::AlignCorners: {
        if (dst_size <= 1) { return 0.0f; }
        return static_cast<float>(dst_idx) *
               static_cast<float>(src_size - 1) /
               static_cast<float>(dst_size - 1);
    }
    case CoordinateTransformMode::Asymmetric:
    default: {
        float scale = static_cast<float>(src_size) / static_cast<float>(dst_size);
        return static_cast<float>(dst_idx) * scale;
    }
    }
}

}  // namespace

template <typename T>
void resize_impl(const ResizeAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = ResizeAttributes::spatial_rank(rank);  // 2 or 3

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

    // Input spatial dims
    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    // Output spatial dims
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    const auto* in_ptr  = input.ptr<T>();
    auto* out_ptr = output.ptr<T>();

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t in_ch_stride   = ID * in_d_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = OD * out_d_stride;

    const auto coord_mode = attrs.coord_mode;
    const auto mode = attrs.mode;
    const bool add_to = attrs.add_to;

    // Per-channel compute lambda (N*C parallel)
    const auto compute_channel = [&](int64_t n, int64_t c) {
        const T* in_ch  = in_ptr + n * C * in_ch_stride + c * in_ch_stride;
        T* out_ch = out_ptr + n * C * out_ch_stride + c * out_ch_stride;

        for (int64_t od = 0; od < OD; ++od) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    float result;

                    if (mode == ResizeMode::Nearest) {
                        // ---- Nearest-neighbor ----
                        if (srank == 3) {
                            float src_d = compute_src_coord(od, ID, OD, coord_mode);
                            float src_h = compute_src_coord(oh, IH, OH, coord_mode);
                            float src_w = compute_src_coord(ow, IW, OW, coord_mode);
                            int64_t id = clamp_idx(static_cast<int64_t>(std::round(src_d)), ID);
                            int64_t ih = clamp_idx(static_cast<int64_t>(std::round(src_h)), IH);
                            int64_t iw = clamp_idx(static_cast<int64_t>(std::round(src_w)), IW);
                            result = s_load(&in_ch[id * in_d_stride + ih * in_row_stride + iw]);
                        } else {
                            float src_h = compute_src_coord(oh, IH, OH, coord_mode);
                            float src_w = compute_src_coord(ow, IW, OW, coord_mode);
                            int64_t ih = clamp_idx(static_cast<int64_t>(std::round(src_h)), IH);
                            int64_t iw = clamp_idx(static_cast<int64_t>(std::round(src_w)), IW);
                            result = s_load(&in_ch[ih * in_row_stride + iw]);
                        }
                    } else {
                        // ---- Linear (bilinear / trilinear) ----
                        if (srank == 3) {
                            // Trilinear: 8 neighbors in 3D
                            float src_d = compute_src_coord(od, ID, OD, coord_mode);
                            float src_h = compute_src_coord(oh, IH, OH, coord_mode);
                            float src_w = compute_src_coord(ow, IW, OW, coord_mode);

                            int64_t z0 = clamp_idx(static_cast<int64_t>(std::floor(src_d)), ID);
                            int64_t y0 = clamp_idx(static_cast<int64_t>(std::floor(src_h)), IH);
                            int64_t x0 = clamp_idx(static_cast<int64_t>(std::floor(src_w)), IW);
                            int64_t z1 = clamp_idx(z0 + 1, ID);
                            int64_t y1 = clamp_idx(y0 + 1, IH);
                            int64_t x1 = clamp_idx(x0 + 1, IW);

                            float wz = src_d - std::floor(src_d);
                            float wy = src_h - std::floor(src_h);
                            float wx = src_w - std::floor(src_w);
                            float wz0 = 1.0f - wz;
                            float wy0 = 1.0f - wy;
                            float wx0 = 1.0f - wx;

                            float v000 = s_load(&in_ch[z0 * in_d_stride + y0 * in_row_stride + x0]);
                            float v100 = s_load(&in_ch[z0 * in_d_stride + y0 * in_row_stride + x1]);
                            float v010 = s_load(&in_ch[z0 * in_d_stride + y1 * in_row_stride + x0]);
                            float v110 = s_load(&in_ch[z0 * in_d_stride + y1 * in_row_stride + x1]);
                            float v001 = s_load(&in_ch[z1 * in_d_stride + y0 * in_row_stride + x0]);
                            float v101 = s_load(&in_ch[z1 * in_d_stride + y0 * in_row_stride + x1]);
                            float v011 = s_load(&in_ch[z1 * in_d_stride + y1 * in_row_stride + x0]);
                            float v111 = s_load(&in_ch[z1 * in_d_stride + y1 * in_row_stride + x1]);

                            float r00 = wx0 * v000 + wx * v100;
                            float r01 = wx0 * v001 + wx * v101;
                            float r10 = wx0 * v010 + wx * v110;
                            float r11 = wx0 * v011 + wx * v111;
                            float r0 = wy0 * r00 + wy * r10;
                            float r1 = wy0 * r01 + wy * r11;
                            result = wz0 * r0 + wz * r1;
                        } else {
                            // Bilinear: 4 neighbors in 2D
                            float src_h = compute_src_coord(oh, IH, OH, coord_mode);
                            float src_w = compute_src_coord(ow, IW, OW, coord_mode);

                            int64_t y0 = clamp_idx(static_cast<int64_t>(std::floor(src_h)), IH);
                            int64_t x0 = clamp_idx(static_cast<int64_t>(std::floor(src_w)), IW);
                            int64_t y1 = clamp_idx(y0 + 1, IH);
                            int64_t x1 = clamp_idx(x0 + 1, IW);

                            float wy = src_h - std::floor(src_h);
                            float wx = src_w - std::floor(src_w);
                            float wy0 = 1.0f - wy;
                            float wx0 = 1.0f - wx;

                            float v00 = s_load(&in_ch[y0 * in_row_stride + x0]);
                            float v10 = s_load(&in_ch[y0 * in_row_stride + x1]);
                            float v01 = s_load(&in_ch[y1 * in_row_stride + x0]);
                            float v11 = s_load(&in_ch[y1 * in_row_stride + x1]);

                            result = wy0 * (wx0 * v00 + wx * v10)
                                   + wy  * (wx0 * v01 + wx * v11);
                        }
                    }

                    const int64_t out_idx = od * out_d_stride + oh * out_row_stride + ow;
                    if (add_to) {
                        s_store(&out_ch[out_idx],
                                s_load(&out_ch[out_idx]) + result);
                    } else {
                        s_store(&out_ch[out_idx], result);
                    }
                }
            }
        }
    };

    const int64_t NC = N * C;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, NC,
            [&](int64_t tid) {
                int64_t n = tid / C;
                int64_t c = tid % C;
                compute_channel(n, c);
            });
    } else {
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C; ++c) {
                compute_channel(n, c);
            }
        }
    }
}

void resize_cpu(const ResizeAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        resize_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        resize_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"resize_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
