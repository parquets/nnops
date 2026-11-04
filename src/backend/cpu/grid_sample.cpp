/// @file grid_sample.cpp
/// @brief CPU implementation of 2D/3D grid sampling, planar and channel-last.
///
/// The image tensor is NCHW/NCDHW or NHWC/NDHWC; the grid is always dense f32
/// ([N, (OD,) OH, OW, 2|3]), because its last axis is a coordinate pair/triple,
/// not a channel. Planar dispatches one task per (n, c) and samples one channel
/// plane per task; channel-last dispatches one task per n and keeps the SIMD
/// lane on C, so a pixel interpolates all its channels at once. The grid
/// coordinate is per output position, so zeros padding is all-or-nothing across
/// the channel vector.

#include "nnops/ops/grid_sample.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_grid_sample.hpp"
#include "common/dtype_dispatch.hpp"

namespace nnops::backend::cpu {

namespace k = nnops::kernel;

template <typename T>
void grid_sample_impl(const GridSampleAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const auto& grid  = inputs[1];
    const int64_t irank = input.rank();
    const int64_t srank = GridSampleAttributes::spatial_rank(irank);

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

    const int64_t ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t IH = input.shape(srank);
    const int64_t IW = input.shape(srank + 1);

    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    if (N == 0 || C == 0 || OD == 0 || OH == 0 || OW == 0) { return; }

    auto*       out_ptr  = output.ptr<T>();
    const auto* in_ptr   = input.ptr<T>();
    const auto* grid_ptr = grid.ptr<float>();

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;

    const int64_t coord_dim       = srank;
    const int64_t grid_row_stride = OW * coord_dim;
    const int64_t grid_n_stride   = grid.stride_elems(0);

    const auto mode         = attrs.mode;
    const auto padding_mode = attrs.padding_mode;
    const bool align        = attrs.align_corners;
    const bool add_to       = attrs.add_to;
    const bool nearest      = (mode == GridSampleMode::Nearest);

    if (is_channel_last(input.layout())) {
        const int64_t in_n_stride  = input.stride_elems(0);
        const int64_t out_n_stride = output.stride_elems(0);

        const auto process = [&](int64_t n) {
            const T* in_base  = in_ptr  + n * in_n_stride;
            T*       out_base = out_ptr + n * out_n_stride;
            const float* grid_n = grid_ptr + n * grid_n_stride;

            if (srank == 3) {
                if (nearest) {
                    k::gridsample_nearest_last_3d<T>(
                        out_base, in_base, grid_n, C,
                        ID, IH, IW, OD, OH, OW,
                        in_d_stride, in_row_stride, out_d_stride, out_row_stride,
                        grid_row_stride, align, padding_mode, add_to);
                } else {
                    k::gridsample_last_3d<T>(
                        out_base, in_base, grid_n, C,
                        ID, IH, IW, OD, OH, OW,
                        in_d_stride, in_row_stride, out_d_stride, out_row_stride,
                        grid_row_stride, align, padding_mode, add_to);
                }
            } else {
                if (nearest) {
                    k::gridsample_nearest_last_2d<T>(
                        out_base, in_base, grid_n, C,
                        IH, IW, OH, OW,
                        in_row_stride, out_row_stride,
                        grid_row_stride, align, padding_mode, add_to);
                } else {
                    k::gridsample_last_2d<T>(
                        out_base, in_base, grid_n, C,
                        IH, IW, OH, OW,
                        in_row_stride, out_row_stride,
                        grid_row_stride, align, padding_mode, add_to);
                }
            }
        };

        ctx.cpu.run(0, N, process);
        return;
    }

    const int64_t in_c_stride  = input.stride_elems(1);
    const int64_t out_c_stride = output.stride_elems(1);
    const int64_t in_n_stride  = input.stride_elems(0);
    const int64_t out_n_stride = output.stride_elems(0);

    const auto process = [&](int64_t task) {
        const int64_t n = task / C;
        const int64_t c = task % C;
        const T* in_base  = in_ptr  + n * in_n_stride  + c * in_c_stride;
        T*       out_base = out_ptr + n * out_n_stride + c * out_c_stride;
        const float* grid_n = grid_ptr + n * grid_n_stride;

        if (srank == 3) {
            if (nearest) {
                k::gridsample_nearest_planar_3d<T>(
                    out_base, in_base, grid_n,
                    ID, IH, IW, OD, OH, OW,
                    in_d_stride, in_row_stride, out_d_stride, out_row_stride,
                    grid_row_stride, align, padding_mode, add_to);
            } else {
                k::gridsample_planar_3d<T>(
                    out_base, in_base, grid_n,
                    ID, IH, IW, OD, OH, OW,
                    in_d_stride, in_row_stride, out_d_stride, out_row_stride,
                    grid_row_stride, align, padding_mode, add_to);
            }
        } else {
            if (nearest) {
                k::gridsample_nearest_planar_2d<T>(
                    out_base, in_base, grid_n,
                    IH, IW, OH, OW,
                    in_row_stride, out_row_stride,
                    grid_row_stride, align, padding_mode, add_to);
            } else {
                k::gridsample_planar_2d<T>(
                    out_base, in_base, grid_n,
                    IH, IW, OH, OW,
                    in_row_stride, out_row_stride,
                    grid_row_stride, align, padding_mode, add_to);
            }
        }
    };

    ctx.cpu.run(0, N * C, process);
}

Status grid_sample_cpu(const GridSampleAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "grid_sample_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        grid_sample_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
