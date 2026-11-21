/// @file grid_sample.cpp
/// @brief CPU implementation of 2D/3D grid sampling on NHWC/NDHWC.
///
/// The image tensor is NHWC/NDHWC; the grid is always dense f32
/// ([N, (OD,) OH, OW, 2|3]), because its last axis is a coordinate pair/triple,
/// not a channel. The dispatch is one task per output row block and keeps the
/// SIMD lane on C, so a pixel interpolates all its channels at once. The grid
/// coordinate is per output position, so zeros padding is all-or-nothing across
/// the channel vector.

#include "nnops/ops/grid_sample.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_grid_sample.hpp"
#include "common/dtype_dispatch.hpp"

#include <algorithm>

namespace nnops::backend::cpu {

namespace k = nnops::kernel;

template <typename T>
void grid_sample_impl(const GridSampleAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& grid = inputs[1];
    const int64_t irank = input.rank();
    const int64_t srank = GridSampleAttributes::spatial_rank(irank);

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(input.rank() - 1);

    const int64_t ID = (srank == 3) ? input.shape(1) : 1;
    const int64_t IH = input.shape(srank - 1);
    const int64_t IW = input.shape(srank);

    const int64_t OD = (srank == 3) ? output.shape(1) : 1;
    const int64_t OH = output.shape(srank - 1);
    const int64_t OW = output.shape(srank);

    if (N == 0 || C == 0 || OD == 0 || OH == 0 || OW == 0) {
        return;
    }

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();
    const auto* grid_ptr = grid.ptr<float>();

    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_d_stride = IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride = OH * out_row_stride;

    const int64_t coord_dim = srank;
    const int64_t grid_row_stride = OW * coord_dim;
    const int64_t grid_n_stride = grid.stride_elems(0);

    const auto mode = attrs.mode;
    const auto padding_mode = attrs.padding_mode;
    const bool align = attrs.align_corners;
    const bool add_to = attrs.add_to;
    const bool nearest = (mode == GridSampleMode::Nearest);

    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t out_n_stride = output.stride_elems(0);

    // One task per output plane (3D) or output row block, not per image: every
    // output row is independent, and a single-image grid (the usual case) would
    // otherwise run the whole sampling loop on one thread. The kernel reads the
    // grid at the row's absolute offset, so a slice takes the matching sub-grid.
    constexpr int64_t kRowsPerTask = 16;
    const int64_t planes = (srank == 3) ? OD : OH;
    const int64_t rows_per_task = std::max<int64_t>(1, std::min(kRowsPerTask, planes));
    const int64_t plane_tiles = (planes + rows_per_task - 1) / rows_per_task;

    const auto process = [&](int64_t task) {
        const int64_t n = task / plane_tiles;
        const int64_t p0 = (task % plane_tiles) * rows_per_task;
        const int64_t pn = std::min(rows_per_task, planes - p0);

        const T* in_base = in_ptr + n * in_n_stride;
        T* out_base = out_ptr + n * out_n_stride;
        const float* grid_n = grid_ptr + n * grid_n_stride;

        if (srank == 3) {
            T* out_d = out_base + p0 * out_d_stride;
            const float* grid_d = grid_n + p0 * OH * grid_row_stride;
            if (nearest) {
                k::gridsample_nearest_last_3d<T>(out_d, in_base, grid_d, C, ID, IH, IW, pn, OH,
                                                 OW, in_d_stride, in_row_stride, out_d_stride,
                                                 out_row_stride, grid_row_stride, align,
                                                 padding_mode, add_to);
            } else {
                k::gridsample_last_3d<T>(out_d, in_base, grid_d, C, ID, IH, IW, pn, OH, OW,
                                         in_d_stride, in_row_stride, out_d_stride, out_row_stride,
                                         grid_row_stride, align, padding_mode, add_to);
            }
        } else {
            T* out_row = out_base + p0 * out_row_stride;
            const float* grid_row = grid_n + p0 * grid_row_stride;
            if (nearest) {
                k::gridsample_nearest_last_2d<T>(out_row, in_base, grid_row, C, IH, IW, pn, OW,
                                                 in_row_stride, out_row_stride, grid_row_stride,
                                                 align, padding_mode, add_to);
            } else {
                k::gridsample_last_2d<T>(out_row, in_base, grid_row, C, IH, IW, pn, OW,
                                         in_row_stride, out_row_stride, grid_row_stride, align,
                                         padding_mode, add_to);
            }
        }
    };

    ctx.cpu.run(0, N * plane_tiles, process);
}

Status grid_sample_cpu(const GridSampleAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx,
                       void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "grid_sample_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        grid_sample_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

} // namespace nnops::backend::cpu
