/// @file resize.cpp
/// @brief SIMD-optimized CPU implementation of 2D/3D spatial resize (NCHWC8/NCDHWC8 only).
///
/// Input/output are always channel-packed: all C8 blocks use SIMD, since the
/// pack step zero-fills a partial block and its pad lanes produce 0 and are
/// never read. Work is dispatched over N*C8 blocks.
///
/// Nearest computes the source coordinate once per output position, then
/// v_loads 8 channels from that position. Bilinear/trilinear compute the
/// coordinate once, load the 4/8 corner vectors and interpolate all 8 lanes
/// with v_fmadd. There is no h4 output-row blocking: resize's output stride
/// differs from the input's, so blocking buys nothing.
///
/// Crop: crop_start/crop_end are indexed as {D,H,W} to match output_size, and
/// crop_end[d] > 0 enables cropping on that axis. The kernel sees the effective
/// size; crop_off (in elements) skips the crop_start region.

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_resize.hpp"
#include "common/dtype_dispatch.hpp"

namespace nnops::backend::cpu {

using namespace nnops::simd;
namespace k = nnops::kernel;

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

    const int64_t raw_ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t raw_IH = input.shape(srank);
    const int64_t raw_IW = input.shape(srank + 1);

    const int64_t pack = input.channel_pack_size();  // 8 for NCHWC8

    const bool has_crop = attrs.has_crop();

    int64_t eff_ID = raw_ID, eff_IH = raw_IH, eff_IW = raw_IW;
    int64_t crop_off = 0;

    if (has_crop) {
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
            crop_off += attrs.crop_start[2] * pack;
        }
    }

    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    auto*       out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();

    // Strides use the raw input dims: the physical layout does not change with crop.
    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = raw_IH * in_row_stride;
    const int64_t in_ch_stride   = input.channel_block_stride_elems();
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;
    const int64_t out_ch_stride  = output.channel_block_stride_elems();

    const auto mode       = attrs.mode;
    const auto coord_mode = attrs.coord_mode;
    const bool add_to     = attrs.add_to;

    const auto compute_c8 = [&](int64_t n, int64_t c8) {
        const T* in_base  = in_ptr  + n * C8 * in_ch_stride  + c8 * in_ch_stride + crop_off;
        T*       out_base = out_ptr + n * C8 * out_ch_stride + c8 * out_ch_stride;

        if (srank == 3) {
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

    const int64_t N_C8 = N * C8;
    ctx.cpu.run(0, N_C8,
            [&](int64_t tid) { compute_c8(tid / C8, tid % C8); });
}

Status resize_cpu(const ResizeAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "resize_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        resize_impl_nchwc8<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
