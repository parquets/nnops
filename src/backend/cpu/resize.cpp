/// @file resize.cpp
/// @brief CPU implementation of 2D/3D spatial resize on NHWC/NDHWC.
///
/// The coordinate tables are built once per call: for each output coordinate,
/// the source index (or the two neighbours and their weight). The kernel
/// interpolates a pixel's whole C vector with SIMD, one sample per task. Cubic
/// swaps the two-tap table for four taps per axis (ResizeCubicTap) and is 2D
/// only.
///
/// Crop: crop_start/crop_end are indexed as {D,H,W} to match output_size, and
/// crop_end[d] > 0 enables cropping on that axis. The kernel sees the effective
/// size; the crop start becomes one element offset through stride_elems().

#include "nnops/ops/resize.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/core/parallel_for.hpp"
#include "simd_kernel/simd_resize.hpp"
#include "common/dtype_dispatch.hpp"

#include <algorithm>
#include <vector>

namespace nnops::backend::cpu {

namespace k = nnops::kernel;

namespace {

struct CoordTable {
    std::vector<int64_t> i0;
    std::vector<int64_t> i1;
    std::vector<float> w;

    void build(int64_t in_size, int64_t out_size, CoordinateTransformMode coord_mode, bool linear)
    {
        i0.resize(static_cast<size_t>(out_size));
        i1.resize(static_cast<size_t>(out_size));
        w.resize(static_cast<size_t>(out_size));
        for (int64_t o = 0; o < out_size; ++o) {
            const float src = k::resize_compute_src_coord(o, in_size, out_size, coord_mode);
            if (!linear) {
                const int64_t idx =
                    k::resize_clamp_idx(static_cast<int64_t>(std::round(src)), in_size);
                i0[static_cast<size_t>(o)] = idx;
                i1[static_cast<size_t>(o)] = idx;
                w[static_cast<size_t>(o)] = 0.0f;
            } else {
                const float floored = std::floor(src);
                const int64_t lo = k::resize_clamp_idx(static_cast<int64_t>(floored), in_size);
                i0[static_cast<size_t>(o)] = lo;
                i1[static_cast<size_t>(o)] = k::resize_clamp_idx(lo + 1, in_size);
                w[static_cast<size_t>(o)] = src - floored;
            }
        }
    }
};

struct CubicTable {
    std::vector<k::ResizeCubicTap> taps;

    void build(int64_t in_size, int64_t out_size, CoordinateTransformMode coord_mode, float a)
    {
        taps.resize(static_cast<size_t>(out_size));
        for (int64_t o = 0; o < out_size; ++o) {
            taps[static_cast<size_t>(o)] = k::resize_make_cubic_tap(
                k::resize_compute_src_coord(o, in_size, out_size, coord_mode), in_size, a);
        }
    }
};

} // anonymous namespace

template <typename T>
void resize_impl(const ResizeAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const auto& input = inputs[0];
    const int64_t rank = input.rank();
    const int64_t srank = ResizeAttributes::spatial_rank(rank);

    const int64_t N = input.shape(0);
    const int64_t C = input.shape(input.rank() - 1);

    // Channel-last dims: [N, (D,) H, W, C].
    const int64_t raw_ID = (srank == 3) ? input.shape(1) : 1;
    const int64_t raw_IH = input.shape(rank - 3);
    const int64_t raw_IW = input.shape(rank - 2);

    int64_t eff_ID = raw_ID, eff_IH = raw_IH, eff_IW = raw_IW;
    int64_t crop_off = 0;

    if (attrs.has_crop()) {
        const int64_t d_dim = (srank == 3) ? 1 : -1;
        const int64_t h_dim = rank - 3;
        const int64_t w_dim = rank - 2;

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

    const int64_t OD = (srank == 3) ? output.shape(1) : 1;
    const int64_t OH = output.shape(rank - 3);
    const int64_t OW = output.shape(rank - 2);

    const bool cubic = (attrs.mode == ResizeMode::Cubic);
    const bool linear = (attrs.mode == ResizeMode::Linear);
    const bool add_to = attrs.add_to;

    CoordTable cx, cy, cz;
    CubicTable kx, ky;
    if (cubic) {
        kx.build(eff_IW, OW, attrs.coord_mode, attrs.cubic_coeff_a);
        ky.build(eff_IH, OH, attrs.coord_mode, attrs.cubic_coeff_a);
    } else {
        cx.build(eff_IW, OW, attrs.coord_mode, linear);
        cy.build(eff_IH, OH, attrs.coord_mode, linear);
        if (srank == 3) {
            cz.build(eff_ID, OD, attrs.coord_mode, linear);
        }
    }

    if (OW == 0 || OH == 0 || OD == 0 || N == 0 || C == 0) {
        return;
    }

    auto* out_ptr = output.ptr<T>();
    const auto* in_ptr = input.ptr<T>();

    const int64_t in_row_stride = input.row_stride_elems();
    const int64_t in_d_stride = raw_IH * in_row_stride;
    const int64_t out_row_stride = output.row_stride_elems();
    const int64_t out_d_stride = OH * out_row_stride;

    const int64_t in_n_stride = input.stride_elems(0);
    const int64_t out_n_stride = output.stride_elems(0);

    // One task per output plane (2D) or output row block: every output row is
    // independent, so a single-image resize — the usual case — still fills the
    // pool. The tables are indexed by the absolute row, so a slice is handed the
    // matching sub-table and a row-offset destination.
    constexpr int64_t kRowsPerTask = 16;
    const int64_t planes = (srank == 3) ? OD : OH;
    const int64_t rows_per_task = std::max<int64_t>(1, std::min(kRowsPerTask, planes));
    const int64_t plane_tiles = (planes + rows_per_task - 1) / rows_per_task;

    const auto process = [&](int64_t task) {
        const int64_t n = task / plane_tiles;
        const int64_t p0 = (task % plane_tiles) * rows_per_task;
        const int64_t pn = std::min(rows_per_task, planes - p0);

        const T* in_base = in_ptr + n * in_n_stride + crop_off;
        T* out_base = out_ptr + n * out_n_stride;

        if (cubic) {
            k::resize_2d_cubic<T>(out_base + p0 * out_row_stride, in_base, C, pn, OW,
                                  ky.taps.data() + p0, kx.taps.data(), in_row_stride,
                                  out_row_stride, add_to);
        } else if (srank == 3) {
            k::resize_3d<T>(out_base + p0 * out_d_stride, in_base, C, pn, OH, OW,
                            cz.i0.data() + p0, cz.i1.data() + p0, cz.w.data() + p0, cy.i0.data(),
                            cy.i1.data(), cy.w.data(), cx.i0.data(), cx.i1.data(), cx.w.data(),
                            in_d_stride, in_row_stride, out_d_stride, out_row_stride, linear,
                            add_to);
        } else {
            k::resize_2d<T>(out_base + p0 * out_row_stride, in_base, C, pn, OW,
                            cy.i0.data() + p0, cy.i1.data() + p0, cy.w.data() + p0, cx.i0.data(),
                            cx.i1.data(), cx.w.data(), in_row_stride, out_row_stride, linear,
                            add_to);
        }
    };

    ctx.cpu.run(0, N * plane_tiles, process);
}

Status resize_cpu(const ResizeAttributes& attrs, TensorView& output,
                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                  void* /*workspace*/)
{
    // Cubic is the 2D bicubic; PyTorch and ONNX Runtime stop there too.
    if (attrs.mode == ResizeMode::Cubic && ResizeAttributes::spatial_rank(inputs[0].rank()) != 2) {
        return Status::unsupported;
    }
    if (!dispatch_f32_f16(inputs[0].data_type(), "resize_cpu", [&](auto tag) {
            using T = typename decltype(tag)::type;
            resize_impl<T>(attrs, output, inputs, ctx);
        })) {
        return Status::unsupported;
    }
    return Status::ok;
}

} // namespace nnops::backend::cpu
