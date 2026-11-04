/// @file resize.cpp
/// @brief CPU implementation of 2D/3D spatial resize, planar and channel-last.
///
/// The coordinate tables are built once per call: for each output coordinate,
/// the source index (or the two neighbours and their weight). Planar walks one
/// channel plane per task and evaluates scalar pixels; channel-last dispatches
/// per sample and lets the kernel interpolate a pixel's whole C vector with SIMD.
///
/// Crop: crop_start/crop_end are indexed as {D,H,W} to match output_size, and
/// crop_end[d] > 0 enables cropping on that axis. The kernel sees the effective
/// size; the crop start becomes one element offset through stride_elems(), so
/// planar and channel-last need no separate arithmetic.

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_resize.hpp"
#include "common/dtype_dispatch.hpp"

#include <vector>

namespace nnops::backend::cpu {

namespace k = nnops::kernel;

namespace {

struct CoordTable {
    std::vector<int64_t> i0;
    std::vector<int64_t> i1;
    std::vector<float>   w;

    void build(int64_t in_size, int64_t out_size,
               CoordinateTransformMode coord_mode, bool linear)
    {
        i0.resize(static_cast<size_t>(out_size));
        i1.resize(static_cast<size_t>(out_size));
        w.resize(static_cast<size_t>(out_size));
        for (int64_t o = 0; o < out_size; ++o) {
            const float src = k::resize_compute_src_coord(o, in_size, out_size, coord_mode);
            if (!linear) {
                const int64_t idx = k::resize_clamp_idx(
                    static_cast<int64_t>(std::round(src)), in_size);
                i0[static_cast<size_t>(o)] = idx;
                i1[static_cast<size_t>(o)] = idx;
                w[static_cast<size_t>(o)]  = 0.0f;
            } else {
                const float floored = std::floor(src);
                const int64_t lo = k::resize_clamp_idx(
                    static_cast<int64_t>(floored), in_size);
                i0[static_cast<size_t>(o)] = lo;
                i1[static_cast<size_t>(o)] = k::resize_clamp_idx(lo + 1, in_size);
                w[static_cast<size_t>(o)]  = src - floored;
            }
        }
    }
};

}  // anonymous namespace

template <typename T>
void resize_impl(const ResizeAttributes& attrs,
                 TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank  = input.rank();
    const int64_t srank = ResizeAttributes::spatial_rank(rank);

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(1);

    const int64_t raw_ID = (srank == 3) ? input.shape(2) : 1;
    const int64_t raw_IH = input.shape(srank);
    const int64_t raw_IW = input.shape(srank + 1);

    int64_t eff_ID = raw_ID, eff_IH = raw_IH, eff_IW = raw_IW;
    int64_t crop_off = 0;

    if (attrs.has_crop()) {
        const int64_t d_dim = (srank == 3) ? 2 : -1;
        const int64_t h_dim = rank - 2;
        const int64_t w_dim = rank - 1;

        if (srank == 3 && attrs.crop_end[0] > 0) {
            NNOPS_ASSERT(attrs.crop_start[0] >= 0 && attrs.crop_start[0] < raw_ID);
            NNOPS_ASSERT(attrs.crop_end[0] > attrs.crop_start[0] && attrs.crop_end[0] <= raw_ID);
            eff_ID = attrs.crop_end[0] - attrs.crop_start[0];
            crop_off += attrs.crop_start[0] * input.stride_elems(d_dim);
        }
        if (attrs.crop_end[1] > 0) {
            NNOPS_ASSERT(attrs.crop_start[1] >= 0 && attrs.crop_start[1] < raw_IH);
            NNOPS_ASSERT(attrs.crop_end[1] > attrs.crop_start[1] && attrs.crop_end[1] <= raw_IH);
            eff_IH = attrs.crop_end[1] - attrs.crop_start[1];
            crop_off += attrs.crop_start[1] * input.stride_elems(h_dim);
        }
        if (attrs.crop_end[2] > 0) {
            NNOPS_ASSERT(attrs.crop_start[2] >= 0 && attrs.crop_start[2] < raw_IW);
            NNOPS_ASSERT(attrs.crop_end[2] > attrs.crop_start[2] && attrs.crop_end[2] <= raw_IW);
            eff_IW = attrs.crop_end[2] - attrs.crop_start[2];
            crop_off += attrs.crop_start[2] * input.stride_elems(w_dim);
        }
    }

    const int64_t OD = (srank == 3) ? output.shape(2) : 1;
    const int64_t OH = output.shape(srank);
    const int64_t OW = output.shape(srank + 1);

    const bool linear = (attrs.mode != ResizeMode::Nearest);
    const bool add_to = attrs.add_to;

    CoordTable cx, cy, cz;
    cx.build(eff_IW, OW, attrs.coord_mode, linear);
    cy.build(eff_IH, OH, attrs.coord_mode, linear);
    if (srank == 3) {
        cz.build(eff_ID, OD, attrs.coord_mode, linear);
    }

    if (OW == 0 || OH == 0 || OD == 0 || N == 0 || C == 0) { return; }

    auto*       out_ptr = output.ptr<T>();
    const auto* in_ptr  = input.ptr<T>();

    const int64_t in_row_stride  = input.row_stride_elems();
    const int64_t in_d_stride    = raw_IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride   = OH * out_row_stride;

    if (is_channel_last(input.layout())) {
        const int64_t in_n_stride  = input.stride_elems(0);
        const int64_t out_n_stride = output.stride_elems(0);

        const auto process = [&](int64_t n) {
            const T* in_base  = in_ptr + n * in_n_stride + crop_off;
            T*       out_base = out_ptr + n * out_n_stride;

            if (srank == 3) {
                k::resize_last_3d<T>(
                    out_base, in_base, C, OD, OH, OW,
                    cz.i0.data(), cz.i1.data(), cz.w.data(),
                    cy.i0.data(), cy.i1.data(), cy.w.data(),
                    cx.i0.data(), cx.i1.data(), cx.w.data(),
                    in_d_stride, in_row_stride, out_d_stride, out_row_stride,
                    linear, add_to);
            } else {
                k::resize_last_2d<T>(
                    out_base, in_base, C, OH, OW,
                    cy.i0.data(), cy.i1.data(), cy.w.data(),
                    cx.i0.data(), cx.i1.data(), cx.w.data(),
                    in_row_stride, out_row_stride, linear, add_to);
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
        const T* in_base  = in_ptr  + n * in_n_stride  + c * in_c_stride  + crop_off;
        T*       out_base = out_ptr + n * out_n_stride + c * out_c_stride;

        if (srank == 3) {
            k::resize_planar_3d<T>(
                out_base, in_base, OD, OH, OW,
                cz.i0.data(), cz.i1.data(), cz.w.data(),
                cy.i0.data(), cy.i1.data(), cy.w.data(),
                cx.i0.data(), cx.i1.data(), cx.w.data(),
                in_d_stride, in_row_stride, out_d_stride, out_row_stride,
                linear, add_to);
        } else {
            k::resize_planar_2d<T>(
                out_base, in_base, OH, OW,
                cy.i0.data(), cy.i1.data(), cy.w.data(),
                cx.i0.data(), cx.i1.data(), cx.w.data(),
                in_row_stride, out_row_stride, linear, add_to);
        }
    };

    ctx.cpu.run(0, N * C, process);
}

Status resize_cpu(const ResizeAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* /*workspace*/)
{
    dispatch_f32_f16(inputs[0].data_type(), "resize_cpu", [&](auto tag) {
        using T = typename decltype(tag)::type;
        resize_impl<T>(attrs, output, inputs, ctx);
    });
    return Status::ok;
}

}  // namespace nnops::backend::cpu
