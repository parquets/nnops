/// @file grid_sample.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D grid sampling (NCHWC8/NCDHWC8 only).
///
/// Input (image to sample) is always NCHWC8 (or NCDHWC8 for 3D). Grid is always
/// plain NCHW/NCDHW float32 (spatial coords, not channel data). All C8 blocks
/// use SIMD — grid coordinates are per-output-position (same for all 8 channels),
/// so validity is all-or-nothing per C8 block.
///
/// Key design decisions:
///   1. Nearest: compute pixel coordinate once per output position, v_load 8
///      channels from source NCHWC8 position, v_store.
///   2. Bilinear/Trilinear: compute pixel coords once, v_load 4/8 corner vectors
///      (8 channels each), interpolate all 8 lanes with v_fmadd (a*b + c).
///   3. Zeros padding: all-or-nothing per C8 block (grid coords are per-position).
///   4. Border/Reflection padding: always valid after padding → SIMD always works.
///   5. N*C8 parallel dispatch.
///   6. Grid access is scalar (grid is NCHW, not channel-packed).

#include "nnops/ops/grid_sample.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;

namespace {

// ============================================================
// Coordinate helpers (identical to scalar version)
// ============================================================

inline float grid_to_pixel(float coord, int64_t size, bool align_corners) {
    if (align_corners) {
        return (coord + 1.0f) * static_cast<float>(size - 1) * 0.5f;
    } else {
        return ((coord + 1.0f) * static_cast<float>(size) - 1.0f) * 0.5f;
    }
}

inline float reflect_coord(float x, int64_t size) {
    if (size <= 1) return 0.0f;
    float max_val = static_cast<float>(size - 1);
    float r = std::fmod(std::abs(x), 2.0f * max_val);
    if (r > max_val) {
        r = 2.0f * max_val - r;
    }
    return r;
}

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

inline bool in_bounds(float pixel, int64_t size) {
    return pixel >= 0.0f && pixel <= static_cast<float>(size - 1);
}

inline int64_t clamp_idx(int64_t idx, int64_t size) {
    return std::max<int64_t>(0, std::min(idx, size - 1));
}

/// Check if all 4 corners of a bilinear region are in bounds.
inline bool bilinear_bounds_ok(float py, float pz, int64_t IH, int64_t IW) {
    return in_bounds(std::floor(py), IH) &&
           in_bounds(std::ceil(py), IH) &&
           in_bounds(std::floor(pz), IW) &&
           in_bounds(std::ceil(pz), IW);
}

/// Check if all 8 corners of a trilinear region are in bounds.
inline bool trilinear_bounds_ok(float px, float py, float pz,
                                 int64_t ID, int64_t IH, int64_t IW) {
    return in_bounds(std::floor(px), ID) &&
           in_bounds(std::ceil(px), ID) &&
           in_bounds(std::floor(py), IH) &&
           in_bounds(std::ceil(py), IH) &&
           in_bounds(std::floor(pz), IW) &&
           in_bounds(std::ceil(pz), IW);
}

// ============================================================
// 2D Bilinear NCHWC8 SIMD kernel
// ============================================================

template <typename T>
void grid_sample_bilinear_2d_nchwc8(
    T* output, const T* input, const float* grid_n,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;  // 2D: (y, x)
            float py = grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = grid_to_pixel(g_pos[1], IW, align_corners);

            if (zeros_pad && !bilinear_bounds_ok(py, pz, IH, IW)) {
                if (!add_to) {
                    v_store(output + ow * 8, v_zero(type_tag));
                }
                continue;
            }

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

            auto v_wy  = v_set1(type_tag, wy);
            auto v_wy0 = v_set1(type_tag, wy0);
            auto v_wz  = v_set1(type_tag, wz);
            auto v_wz0 = v_set1(type_tag, wz0);

            // Load 4 corner vectors (8 channels each)
            auto v00 = v_load(&input[y0 * in_row_stride + z0 * 8]);
            auto v01 = v_load(&input[y0 * in_row_stride + z1 * 8]);
            auto v10 = v_load(&input[y1 * in_row_stride + z0 * 8]);
            auto v11 = v_load(&input[y1 * in_row_stride + z1 * 8]);

            // result = wy0*(wz0*v00 + wz*v01) + wy*(wz0*v10 + wz*v11)
            auto row0 = v_fmadd(v_wz0, v00, v_mul(v_wz, v01));
            auto row1 = v_fmadd(v_wz0, v10, v_mul(v_wz, v11));
            auto result = v_fmadd(v_wy0, row0, v_mul(v_wy, row1));

            v_store_add(output + ow * 8, result, add_to);
        }
        output += out_row_stride;
    }
}

// ============================================================
// 2D Nearest-neighbor NCHWC8 SIMD kernel
// ============================================================

template <typename T>
void grid_sample_nearest_2d_nchwc8(
    T* output, const T* input, const float* grid_n,
    int64_t IH, int64_t IW, int64_t OH, int64_t OW,
    int64_t in_row_stride, int64_t out_row_stride,
    int64_t grid_row_stride,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t oh = 0; oh < OH; ++oh) {
        const float* g_row = grid_n + oh * grid_row_stride;

        for (int64_t ow = 0; ow < OW; ++ow) {
            const float* g_pos = g_row + ow * 2;
            float py = grid_to_pixel(g_pos[0], IH, align_corners);
            float pz = grid_to_pixel(g_pos[1], IW, align_corners);

            float py_r = std::round(py);
            float pz_r = std::round(pz);

            if (zeros_pad && (!in_bounds(py_r, IH) || !in_bounds(pz_r, IW))) {
                if (!add_to) {
                    v_store(output + ow * 8, v_zero(type_tag));
                }
                continue;
            }

            py = apply_padding(py, IH, padding);
            pz = apply_padding(pz, IW, padding);
            int64_t iy = static_cast<int64_t>(std::round(py));
            int64_t iz = static_cast<int64_t>(std::round(pz));

            auto val = v_load(&input[iy * in_row_stride + iz * 8]);
            v_store_add(output + ow * 8, val, add_to);
        }
        output += out_row_stride;
    }
}

// ============================================================
// 3D Trilinear NCDHWC8 SIMD kernel
// ============================================================

template <typename T>
void grid_sample_trilinear_3d_ncdhwc8(
    T* output, const T* input, const float* grid_n,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride, int64_t grid_d_elems,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t od = 0; od < OD; ++od) {
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const float* g_pos = grid_n + (od * OH + oh) * grid_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float px = grid_to_pixel(g_pos[0], ID, align_corners);
                float py = grid_to_pixel(g_pos[1], IH, align_corners);
                float pz = grid_to_pixel(g_pos[2], IW, align_corners);
                g_pos += 3;  // coord_dim = 3

                if (zeros_pad && !trilinear_bounds_ok(px, py, pz, ID, IH, IW)) {
                    if (!add_to) {
                        v_store(out_d + oh * out_row_stride + ow * 8, v_zero(type_tag));
                    }
                    continue;
                }

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

                auto v_wx  = v_set1(type_tag, wx);
                auto v_wx0 = v_set1(type_tag, wx0);
                auto v_wy  = v_set1(type_tag, wy);
                auto v_wy0 = v_set1(type_tag, wy0);
                auto v_wz  = v_set1(type_tag, wz);
                auto v_wz0 = v_set1(type_tag, wz0);

                // Load 8 corner vectors
                auto v000 = v_load(&input[x0 * in_d_stride + y0 * in_row_stride + z0 * 8]);
                auto v001 = v_load(&input[x0 * in_d_stride + y0 * in_row_stride + z1 * 8]);
                auto v010 = v_load(&input[x0 * in_d_stride + y1 * in_row_stride + z0 * 8]);
                auto v011 = v_load(&input[x0 * in_d_stride + y1 * in_row_stride + z1 * 8]);
                auto v100 = v_load(&input[x1 * in_d_stride + y0 * in_row_stride + z0 * 8]);
                auto v101 = v_load(&input[x1 * in_d_stride + y0 * in_row_stride + z1 * 8]);
                auto v110 = v_load(&input[x1 * in_d_stride + y1 * in_row_stride + z0 * 8]);
                auto v111 = v_load(&input[x1 * in_d_stride + y1 * in_row_stride + z1 * 8]);

                // Interpolate across z, then y, then x
                auto r00 = v_fmadd(v_wz0, v000, v_mul(v_wz, v001));
                auto r01 = v_fmadd(v_wz0, v010, v_mul(v_wz, v011));
                auto r10 = v_fmadd(v_wz0, v100, v_mul(v_wz, v101));
                auto r11 = v_fmadd(v_wz0, v110, v_mul(v_wz, v111));

                auto s0 = v_fmadd(v_wy0, r00, v_mul(v_wy, r01));
                auto s1 = v_fmadd(v_wy0, r10, v_mul(v_wy, r11));

                auto result = v_fmadd(v_wx0, s0, v_mul(v_wx, s1));

                v_store_add(out_d + oh * out_row_stride + ow * 8, result, add_to);
            }
        }
    }
}

// ============================================================
// 3D Nearest-neighbor NCDHWC8 SIMD kernel
// ============================================================

template <typename T>
void grid_sample_nearest_3d_ncdhwc8(
    T* output, const T* input, const float* grid_n,
    int64_t ID, int64_t IH, int64_t IW,
    int64_t OD, int64_t OH, int64_t OW,
    int64_t in_d_stride, int64_t in_row_stride,
    int64_t out_d_stride, int64_t out_row_stride,
    int64_t grid_row_stride, int64_t grid_d_elems,
    bool align_corners, GridSamplePaddingMode padding, bool add_to)
{
    const T* type_tag = output;
    const bool zeros_pad = (padding == GridSamplePaddingMode::Zeros);

    for (int64_t od = 0; od < OD; ++od) {
        T* out_d = output + od * out_d_stride;

        for (int64_t oh = 0; oh < OH; ++oh) {
            const float* g_pos = grid_n + (od * OH + oh) * grid_row_stride;

            for (int64_t ow = 0; ow < OW; ++ow) {
                float px = grid_to_pixel(g_pos[0], ID, align_corners);
                float py = grid_to_pixel(g_pos[1], IH, align_corners);
                float pz = grid_to_pixel(g_pos[2], IW, align_corners);
                g_pos += 3;

                float px_r = std::round(px);
                float py_r = std::round(py);
                float pz_r = std::round(pz);

                if (zeros_pad &&
                    (!in_bounds(px_r, ID) || !in_bounds(py_r, IH) || !in_bounds(pz_r, IW))) {
                    if (!add_to) {
                        v_store(out_d + oh * out_row_stride + ow * 8, v_zero(type_tag));
                    }
                    continue;
                }

                px = apply_padding(px, ID, padding);
                py = apply_padding(py, IH, padding);
                pz = apply_padding(pz, IW, padding);
                int64_t ix = static_cast<int64_t>(std::round(px));
                int64_t iy = static_cast<int64_t>(std::round(py));
                int64_t iz = static_cast<int64_t>(std::round(pz));

                auto val = v_load(&input[ix * in_d_stride + iy * in_row_stride + iz * 8]);
                v_store_add(out_d + oh * out_row_stride + ow * 8, val, add_to);
            }
        }
    }
}

}  // anonymous namespace

// ============================================================
// Main grid_sample implementation — N*C8 parallel dispatch
// ============================================================

template <typename T>
void grid_sample_impl_nchwc8(const GridSampleAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& grid  = inputs[1];
    const int64_t irank = input.rank();
    const int64_t srank = GridSampleAttributes::spatial_rank(irank);

    const int64_t N  = input.shape(0);
    const int64_t C8 = input.num_channel_blocks();

    // Input spatial dims
    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    // Output spatial dims (from grid)
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    auto*       out_ptr  = output.ptr<T>();
    const auto* in_ptr   = input.ptr<T>();
    const auto* grid_ptr = grid.ptr<float>();

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t in_ch_stride   = input.channel_block_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = output.channel_block_stride_elems();

    // Grid strides (grid is plain NCHW/NCDHW float32, not channel-packed)
    const int64_t coord_dim = srank;
    const int64_t grid_row_stride = OW * coord_dim;
    const int64_t grid_d_elems    = OH * grid_row_stride;

    const auto mode         = attrs.mode;
    const auto padding_mode = attrs.padding_mode;
    const bool align        = attrs.align_corners;
    const bool add_to       = attrs.add_to;

    // Per-C8-block compute lambda (all C8 blocks use SIMD)
    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const T* in_base  = in_ptr  + n * C8 * in_ch_stride  + c8 * in_ch_stride;
        T*       out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;

        // Grid for this sample (NCHW, not channel-packed)
        const float* grid_n = (srank == 3)
            ? grid_ptr + n * OD * grid_d_elems
            : grid_ptr + n * OH * grid_row_stride;

        if (srank == 3) {
            if (mode == GridSampleMode::Nearest) {
                grid_sample_nearest_3d_ncdhwc8<T>(
                    out_base, in_base, grid_n,
                    ID, IH, IW, OD, OH, OW,
                    in_d_stride, in_row_stride,
                    out_d_stride, out_row_stride,
                    grid_row_stride, grid_d_elems,
                    align, padding_mode, add_to);
            } else {
                grid_sample_trilinear_3d_ncdhwc8<T>(
                    out_base, in_base, grid_n,
                    ID, IH, IW, OD, OH, OW,
                    in_d_stride, in_row_stride,
                    out_d_stride, out_row_stride,
                    grid_row_stride, grid_d_elems,
                    align, padding_mode, add_to);
            }
        } else {
            if (mode == GridSampleMode::Nearest) {
                grid_sample_nearest_2d_nchwc8<T>(
                    out_base, in_base, grid_n,
                    IH, IW, OH, OW,
                    in_row_stride, out_row_stride,
                    grid_row_stride,
                    align, padding_mode, add_to);
            } else {
                grid_sample_bilinear_2d_nchwc8<T>(
                    out_base, in_base, grid_n,
                    IH, IW, OH, OW,
                    in_row_stride, out_row_stride,
                    grid_row_stride,
                    align, padding_mode, add_to);
            }
        }
    };

    // Parallel dispatch (N * C8)
    const int64_t N_C8 = N * C8;
    if (ctx.cpu_parallel_for) {
        ctx.cpu_parallel_for(0, N_C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
    } else {
        for (int64_t n = 0; n < N; ++n)
            for (int64_t c8 = 0; c8 < C8; ++c8)
                compute_c8(n, c8);
    }
}

// ============================================================
// Entry point — dtype dispatch
// ============================================================

void grid_sample_cpu(const GridSampleAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        grid_sample_impl_nchwc8<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        grid_sample_impl_nchwc8<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"grid_sample_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
