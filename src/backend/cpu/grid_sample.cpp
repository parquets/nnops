/// @file grid_sample.cpp
/// @brief CPU implementation of 2D/3D grid sampling (NCHW/NCDHW layout).
///
/// Supports both f32 and f16 data types. Grid coordinates are always f32.
/// Computation always done in fp32; dtype-specific load/store handled via
/// s_load/s_store helpers.
///
/// Implements bilinear (2D) and nearest-neighbor interpolation with three
/// padding modes: zeros, border (clamp), and reflection.
///
/// Reference: ONNX GridSample operator (since opset 16).
///
/// Parallelism is over N*C tasks (sample x channel), allowing better thread
/// utilization especially for small-batch inference (N=1).

#include "nnops/ops/grid_sample.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

/// Map a normalized grid coordinate to a pixel index (as float).
/// align_corners=true:  x_pixel = (coord + 1) * (size - 1) / 2
/// align_corners=false: x_pixel = ((coord + 1) * size - 1) / 2
inline float grid_to_pixel(float coord, int64_t size, bool align_corners) {
    if (align_corners) {
        return (coord + 1.0f) * static_cast<float>(size - 1) * 0.5f;
    } else {
        return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
    }
}

/// Reflect coordinates at boundaries: mirrors at 0 and size-1.
/// e.g. for size=4, valid [0,3]: reflect(-1)=1, reflect(-2)=2, reflect(4)=2, reflect(5)=1
inline float reflect_coord(float x, int64_t size) {
    if (size <= 1) return 0.0f;
    float max_val = static_cast<float>(size - 1);
    float r = std::fmod(std::abs(x), 2.0f * max_val);
    if (r > max_val) {
        r = 2.0f * max_val - r;
    }
    return r;
}

/// Apply padding mode to a pixel coordinate, returning the final float coordinate.
inline float apply_padding(float pixel, int64_t size, GridSamplePaddingMode padding) {
    switch (padding) {
    case GridSamplePaddingMode::Border:
        return std::max(0.0f, std::min(pixel, static_cast<float>(size - 1)));
    case GridSamplePaddingMode::Reflection:
        return reflect_coord(pixel, size);
    case GridSamplePaddingMode::Zeros:
    default:
        return pixel;
    }
}

/// Check if a pixel coordinate is within bounds.
inline bool in_bounds(float pixel, int64_t size) {
    return pixel >= 0.0f && pixel <= static_cast<float>(size - 1);
}

/// Clamp an integer index to [0, size-1].
inline int64_t clamp_idx(int64_t idx, int64_t size) {
    return std::max<int64_t>(0, std::min(idx, size - 1));
}

}  // namespace

template <typename T>
void grid_sample_impl(const GridSampleAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& grid  = inputs[1];
    const int64_t irank = input.rank();
    const int64_t srank = GridSampleAttributes::spatial_rank(irank);  // 2 or 3

    const int64_t N  = input.shape(0);
    const int64_t C  = input.shape(1);

    // Input spatial dims
    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    // Output spatial dims (from grid)
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    const auto* in_ptr = input.ptr<T>();
    const auto* grid_ptr = grid.ptr<float>();  // Grid is always float32
    auto* out_ptr = output.ptr<T>();

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t in_ch_stride   = ID * in_d_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = OD * out_d_stride;

    // Grid stride: grid shape = [N, (OD,) OH, OW, coord_dim]
    // coord_dim = srank + 1 (2D → 3: [y, x, unused], 3D → 4: [x, y, z, unused]
    // Actually ONNX convention: grid last dim = spatial_rank
    // For 2D: grid [N, OH, OW, 2] → coord_dim = 2
    // For 3D: grid [N, OD, OH, OW, 3] → coord_dim = 3
    const int64_t coord_dim = srank;  // 2 or 3
    const int64_t grid_last_dim_elems = OW * coord_dim;  // elements per row in grid
    const int64_t grid_d_elems = OH * grid_last_dim_elems;

    const auto mode = attrs.mode;
    const auto padding = attrs.padding_mode;
    const bool align = attrs.align_corners;
    const bool add_to = attrs.add_to;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    // Per-channel compute lambda (N*C parallel)
    const auto compute_channel = [&](int64_t n, int64_t c) {
        const T* in_ch  = in_ptr + n * C * in_ch_stride + c * in_ch_stride;
        T* out_ch = out_ptr + n * C * out_ch_stride + c * out_ch_stride;

        // Grid for this sample
        const float* grid_n = (srank == 3)
            ? grid_ptr + n * OD * grid_d_elems
            : grid_ptr + n * OH * grid_last_dim_elems;

        for (int64_t od = 0; od < OD; ++od) {
            for (int64_t oh = 0; oh < OH; ++oh) {
                for (int64_t ow = 0; ow < OW; ++ow) {
                    // Read grid coordinates
                    const float* g_pos = (srank == 3)
                        ? grid_n + (od * OH + oh) * grid_last_dim_elems + ow * coord_dim
                        : grid_n + oh * grid_last_dim_elems + ow * coord_dim;

                    float result = 0.0f;
                    bool valid = true;

                    if (srank == 3) {
                        // ---- 3D (trilinear or nearest-neighbor) ----
                        float px = grid_to_pixel(g_pos[0], ID, align);
                        float py = grid_to_pixel(g_pos[1], IH, align);
                        float pz = grid_to_pixel(g_pos[2], IW, align);

                        if (mode == GridSampleMode::Nearest) {
                            if (zeros_pad &&
                                (!in_bounds(std::round(px), ID) ||
                                 !in_bounds(std::round(py), IH) ||
                                 !in_bounds(std::round(pz), IW))) {
                                valid = false;
                            } else {
                                px = apply_padding(px, ID, padding);
                                py = apply_padding(py, IH, padding);
                                pz = apply_padding(pz, IW, padding);
                                int64_t ix = static_cast<int64_t>(std::round(px));
                                int64_t iy = static_cast<int64_t>(std::round(py));
                                int64_t iz = static_cast<int64_t>(std::round(pz));
                                result = s_load(&in_ch[ix * in_d_stride +
                                                      iy * in_row_stride + iz]);
                            }
                        } else {
                            // Trilinear
                            if (zeros_pad &&
                                (!in_bounds(std::floor(px), ID) ||
                                 !in_bounds(std::ceil(px), ID) ||
                                 !in_bounds(std::floor(py), IH) ||
                                 !in_bounds(std::ceil(py), IH) ||
                                 !in_bounds(std::floor(pz), IW) ||
                                 !in_bounds(std::ceil(pz), IW))) {
                                valid = false;
                            } else {
                                px = apply_padding(px, ID, padding);
                                py = apply_padding(py, IH, padding);
                                pz = apply_padding(pz, IW, padding);

                                int64_t x0 = static_cast<int64_t>(std::floor(px));
                                int64_t y0 = static_cast<int64_t>(std::floor(py));
                                int64_t z0 = static_cast<int64_t>(std::floor(pz));
                                int64_t x1 = clamp_idx(x0 + 1, ID);
                                int64_t y1 = clamp_idx(y0 + 1, IH);
                                int64_t z1 = clamp_idx(z0 + 1, IW);

                                float wx = px - static_cast<float>(x0);
                                float wy = py - static_cast<float>(y0);
                                float wz = pz - static_cast<float>(z0);
                                float wx0 = 1.0f - wx;
                                float wy0 = 1.0f - wy;
                                float wz0 = 1.0f - wz;

                                result =
                                    wx0 * wy0 * wz0 * s_load(&in_ch[x0 * in_d_stride + y0 * in_row_stride + z0]) +
                                    wx  * wy0 * wz0 * s_load(&in_ch[x1 * in_d_stride + y0 * in_row_stride + z0]) +
                                    wx0 * wy  * wz0 * s_load(&in_ch[x0 * in_d_stride + y1 * in_row_stride + z0]) +
                                    wx  * wy  * wz0 * s_load(&in_ch[x1 * in_d_stride + y1 * in_row_stride + z0]) +
                                    wx0 * wy0 * wz  * s_load(&in_ch[x0 * in_d_stride + y0 * in_row_stride + z1]) +
                                    wx  * wy0 * wz  * s_load(&in_ch[x1 * in_d_stride + y0 * in_row_stride + z1]) +
                                    wx0 * wy  * wz  * s_load(&in_ch[x0 * in_d_stride + y1 * in_row_stride + z1]) +
                                    wx  * wy  * wz  * s_load(&in_ch[x1 * in_d_stride + y1 * in_row_stride + z1]);
                            }
                        }

                        const int64_t out_idx = od * out_d_stride + oh * out_row_stride + ow;
                        if (valid) {
                            if (add_to) {
                                s_store(&out_ch[out_idx], s_load(&out_ch[out_idx]) + result);
                            } else {
                                s_store(&out_ch[out_idx], result);
                            }
                        } else {
                            if (!add_to) {
                                s_store(&out_ch[out_idx], 0.0f);
                            }
                            // add_to with zeros pad: no change to output
                        }
                    } else {
                        // ---- 2D (bilinear or nearest-neighbor) ----
                        float py = grid_to_pixel(g_pos[0], IH, align);
                        float pz = grid_to_pixel(g_pos[1], IW, align);

                        if (mode == GridSampleMode::Nearest) {
                            if (zeros_pad &&
                                (!in_bounds(std::round(py), IH) ||
                                 !in_bounds(std::round(pz), IW))) {
                                valid = false;
                            } else {
                                py = apply_padding(py, IH, padding);
                                pz = apply_padding(pz, IW, padding);
                                int64_t iy = static_cast<int64_t>(std::round(py));
                                int64_t iz = static_cast<int64_t>(std::round(pz));
                                result = s_load(&in_ch[iy * in_row_stride + iz]);
                            }
                        } else {
                            // Bilinear
                            if (zeros_pad &&
                                (!in_bounds(std::floor(py), IH) ||
                                 !in_bounds(std::ceil(py), IH) ||
                                 !in_bounds(std::floor(pz), IW) ||
                                 !in_bounds(std::ceil(pz), IW))) {
                                valid = false;
                            } else {
                                py = apply_padding(py, IH, padding);
                                pz = apply_padding(pz, IW, padding);

                                int64_t y0 = static_cast<int64_t>(std::floor(py));
                                int64_t z0 = static_cast<int64_t>(std::floor(pz));
                                int64_t y1 = clamp_idx(y0 + 1, IH);
                                int64_t z1 = clamp_idx(z0 + 1, IW);

                                float wy = py - static_cast<float>(y0);
                                float wz = pz - static_cast<float>(z0);
                                float wy0 = 1.0f - wy;
                                float wz0 = 1.0f - wz;

                                float v00 = s_load(&in_ch[y0 * in_row_stride + z0]);
                                float v01 = s_load(&in_ch[y0 * in_row_stride + z1]);
                                float v10 = s_load(&in_ch[y1 * in_row_stride + z0]);
                                float v11 = s_load(&in_ch[y1 * in_row_stride + z1]);

                                result = wy0 * (wz0 * v00 + wz * v01)
                                       + wy  * (wz0 * v10 + wz * v11);
                            }
                        }

                        const int64_t out_idx = oh * out_row_stride + ow;
                        if (valid) {
                            if (add_to) {
                                s_store(&out_ch[out_idx], s_load(&out_ch[out_idx]) + result);
                            } else {
                                s_store(&out_ch[out_idx], result);
                            }
                        } else {
                            if (!add_to) {
                                s_store(&out_ch[out_idx], 0.0f);
                            }
                        }
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

void grid_sample_cpu(const GridSampleAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        grid_sample_impl<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        grid_sample_impl<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"grid_sample_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
