/// @file resize.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial resize (NCHWC8/NCDHWC8 only).
///
/// Input/output are always channel-packed (NCHWC8 or NCDHWC8). All C8 blocks
/// use SIMD — partial C8 blocks are zero-padded by the pack step, so all 8
/// lanes compute correctly (pad lanes produce 0 results and are never read).
///
/// Key design decisions:
///   1. Nearest: compute source coordinate once per output position, then
///      v_load 8 channels at once from the source NCHWC8 position.
///   2. Bilinear/Trilinear: compute source coordinates once, load 4/8 corner
///      vectors (8 channels each), interpolate all 8 lanes in parallel via
///      v_fmadd (a*b + c).
///   3. N*C8 parallel dispatch.
///   4. No h4 blocking — resize doesn't benefit from output-row blocking
///      (output stride differs from input stride).

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "nnops/core/tensor_layout.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_resize.hpp"

#include <algorithm>
#include <cmath>

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

// ============================================================
// Main resize implementation — N*C8 parallel dispatch
// ============================================================

template <typename T>
void resize_impl_nchwc8(const ResizeAttributes& attrs,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t srank = ResizeAttributes::spatial_rank(rank);

    const int64_t N   = input.shape(0);
    const int64_t C8  = input.num_channel_blocks();

    // Input spatial dims
    const int64_t raw_ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t raw_IH = input.shape(srank);
    const int64_t raw_IW = input.shape(srank + 1);

    // ---- Crop: compute effective input region ----
    // When crop_end[d] > 0, use crop region; otherwise use full input.
    // crop_start / crop_end use {D, H, W} indexing matching output_size.
    const int64_t pack = input.channel_pack_size();  // 8 for NCHWC8

    const bool has_crop = attrs.has_crop();

    // Effective source sizes (what the kernel sees as its "input")
    int64_t eff_ID = raw_ID, eff_IH = raw_IH, eff_IW = raw_IW;
    // Offset added to in_base to skip crop_start region (in elements)
    int64_t crop_off = 0;

    if (has_crop) {
        // Validate crop bounds
        if (srank == 3 && attrs.crop_end[0] > 0) {
            NNOPS_ASSERT(attrs.crop_start[0] >= 0 && attrs.crop_start[0] < raw_ID);
            NNOPS_ASSERT(attrs.crop_end[0] > attrs.crop_start[0] && attrs.crop_end[0] <= raw_ID);
            eff_ID = attrs.crop_end[0] - attrs.crop_start[0];
            crop_off += attrs.crop_start[0] * raw_IH * input.row_stride_elems();
        }
        if (attrs.crop_end[1] > 0) {
            NNOPS_ASSERT(attrs.crop_start[1] >= 0 && attrs.crop_start[1] < raw_IH);
            NNOPS_ASSERT(attrs.crop_end[1] > attrs.crop_start[1] && attrs.crop_end[1] <= raw_IH);
            eff_IH = attrs.crop_end[1] - attrs.crop_start[1];
            crop_off += attrs.crop_start[1] * input.row_stride_elems();
        }
        if (attrs.crop_end[2] > 0) {
            NNOPS_ASSERT(attrs.crop_start[2] >= 0 && attrs.crop_start[2] < raw_IW);
            NNOPS_ASSERT(attrs.crop_end[2] > attrs.crop_start[2] && attrs.crop_end[2] <= raw_IW);
            eff_IW = attrs.crop_end[2] - attrs.crop_start[2];
            crop_off += attrs.crop_start[2] * pack;  // W pixels × 8 channels
        }
    }

    // Output spatial dims
    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    auto*       out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();

    // Strides use raw input dims (physical layout doesn't change with crop)
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = raw_IH * in_row_stride;
    const int64_t in_ch_stride   = input.channel_block_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = output.channel_block_stride_elems();

    const auto mode       = attrs.mode;
    const auto coord_mode = attrs.coord_mode;
    const bool add_to     = attrs.add_to;

    // Per-C8-block compute lambda (all C8 blocks use SIMD)
    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const T* in_base  = in_ptr  + n * C8 * in_ch_stride  + c8 * in_ch_stride + crop_off;
        T*       out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;

        if (srank == 3) {
            // 3D
            if (mode == ResizeMode::Nearest) {
                k::resize_nearest_3d<T>(
                    out_base, in_base,
                    eff_ID, eff_IH, eff_IW, OD, OH, OW,
                    in_d_stride, in_row_stride,
                    out_d_stride, out_row_stride,
                    coord_mode, add_to);
            } else {
                k::resize_trilinear_3d<T>(
                    out_base, in_base,
                    eff_ID, eff_IH, eff_IW, OD, OH, OW,
                    in_d_stride, in_row_stride,
                    out_d_stride, out_row_stride,
                    coord_mode, add_to);
            }
        } else {
            // 2D
            if (mode == ResizeMode::Nearest) {
                k::resize_nearest_2d<T>(
                    out_base, in_base,
                    eff_IH, eff_IW, OH, OW,
                    in_row_stride, out_row_stride,
                    coord_mode, add_to);
            } else {
                k::resize_bilinear_2d<T>(
                    out_base, in_base,
                    eff_IH, eff_IW, OH, OW,
                    in_row_stride, out_row_stride,
                    coord_mode, add_to);
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

void resize_cpu(const ResizeAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    const auto dtype = inputs[0].data_type();
    switch (dtype) {
    case DataType::f32:
        resize_impl_nchwc8<float>(attrs, output, inputs, ctx);
        return;
    case DataType::f16:
        resize_impl_nchwc8<half>(attrs, output, inputs, ctx);
        return;
    default:
        NNOPS_ASSERT(!"resize_cpu: unsupported data type (only f32 and f16)");
    }
}

}  // namespace nnops::backend::cpu
