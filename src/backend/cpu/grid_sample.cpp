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
#include "simd_kernel/simd_grid_sample.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

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
                k::gridsample_nearest_3d<T>(
                    out_base, in_base, grid_n,
                    ID, IH, IW, OD, OH, OW,
                    in_d_stride, in_row_stride,
                    out_d_stride, out_row_stride,
                    grid_row_stride, grid_d_elems,
                    align, padding_mode, add_to);
            } else {
                k::gridsample_trilinear_3d<T>(
                    out_base, in_base, grid_n,
                    ID, IH, IW, OD, OH, OW,
                    in_d_stride, in_row_stride,
                    out_d_stride, out_row_stride,
                    grid_row_stride, grid_d_elems,
                    align, padding_mode, add_to);
            }
        } else {
            if (mode == GridSampleMode::Nearest) {
                k::gridsample_nearest_2d<T>(
                    out_base, in_base, grid_n,
                    IH, IW, OH, OW,
                    in_row_stride, out_row_stride,
                    grid_row_stride,
                    align, padding_mode, add_to);
            } else {
                k::gridsample_bilinear_2d<T>(
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
    ctx.cpu.run(0, N_C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
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
