/// @file layout_convert.cpp
/// @brief CPU implementation of planar ↔ channel-last conversion.
///
/// Both directions walk one spatial row at a time (W pixels, C channels) and
/// hand 8-channel blocks to the transpose kernel. 2D and 3D share the loop:
/// only the number of spatial rows differs (H vs D·H). Parallelism flattens
/// N × spatial-rows into one task grid.

#include "backend/cpu/layout_convert.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "simd_kernel/simd_layout_convert.hpp"

#include <algorithm>
#include <cstdint>

namespace nnops {

using namespace nnops::simd;
namespace k = nnops::kernel;

namespace {

template <typename T>
void planar_to_cl_impl(const TensorView& src, TensorView& dst,
                       const ComputeContext& ctx)
{
    const int64_t rank = src.rank();
    const int64_t N = src.shape(0);
    const int64_t C = src.shape(1);
    const int64_t W = src.shape(rank - 1);

    int64_t spatial_rows = 1;
    for (int64_t d = 2; d < rank - 1; ++d) {
        spatial_rows *= src.shape(d);
    }

    const int64_t in_ch_stride   = src.stride_elems(1);
    const int64_t in_row_stride  = src.row_stride_elems();
    const int64_t in_n_stride    = src.stride_elems(0);
    const int64_t out_row_stride = dst.row_stride_elems();
    const int64_t out_n_stride   = dst.stride_elems(0);

    const T* in_ptr  = src.ptr<T>();
    T*       out_ptr = dst.ptr<T>();

    const auto process = [&](int64_t task) {
        const int64_t n  = task / spatial_rows;
        const int64_t sr = task % spatial_rows;

        const T* in_row  = in_ptr  + n * in_n_stride  + sr * in_row_stride;
        T*       out_row = out_ptr + n * out_n_stride + sr * out_row_stride;

        for (int64_t c_base = 0; c_base < C; c_base += 8) {
            const int64_t valid = std::min<int64_t>(8, C - c_base);
            k::layout_planar_to_cl_block<T>(in_row, out_row, W, C, in_ch_stride,
                                            c_base, valid);
        }
    };

    ctx.cpu.run(0, N * spatial_rows, process);
}

template <typename T>
void cl_to_planar_impl(const TensorView& src, TensorView& dst,
                       const ComputeContext& ctx)
{
    const int64_t rank = src.rank();
    const int64_t N = src.shape(0);
    const int64_t C = src.shape(1);
    const int64_t W = src.shape(rank - 1);

    int64_t spatial_rows = 1;
    for (int64_t d = 2; d < rank - 1; ++d) {
        spatial_rows *= src.shape(d);
    }

    const int64_t in_row_stride  = src.row_stride_elems();
    const int64_t in_n_stride    = src.stride_elems(0);
    const int64_t out_ch_stride  = dst.stride_elems(1);
    const int64_t out_row_stride = dst.row_stride_elems();
    const int64_t out_n_stride   = dst.stride_elems(0);

    const T* in_ptr  = src.ptr<T>();
    T*       out_ptr = dst.ptr<T>();

    const auto process = [&](int64_t task) {
        const int64_t n  = task / spatial_rows;
        const int64_t sr = task % spatial_rows;

        const T* in_row  = in_ptr  + n * in_n_stride  + sr * in_row_stride;
        T*       out_row = out_ptr + n * out_n_stride + sr * out_row_stride;

        for (int64_t c_base = 0; c_base < C; c_base += 8) {
            const int64_t valid = std::min<int64_t>(8, C - c_base);
            k::layout_cl_to_planar_block<T>(in_row, out_row, W, C, out_ch_stride,
                                            c_base, valid);
        }
    };

    ctx.cpu.run(0, N * spatial_rows, process);
}

}  // anonymous namespace

void convert_planar_to_channel_last(const TensorView& src, TensorView& dst,
                                    const ComputeContext& ctx) {
    NNOPS_ASSERT(src.data_type() == dst.data_type());
    switch (src.data_type()) {
    case DataType::f32: planar_to_cl_impl<float>(src, dst, ctx); return;
    case DataType::f16: planar_to_cl_impl<half>(src, dst, ctx);  return;
    case DataType::s8:  planar_to_cl_impl<int8_t>(src, dst, ctx); return;
    case DataType::u8:  planar_to_cl_impl<uint8_t>(src, dst, ctx); return;
    default: NNOPS_ASSERT(!"convert_planar_to_channel_last: unsupported data type");
    }
}

void convert_channel_last_to_planar(const TensorView& src, TensorView& dst,
                                    const ComputeContext& ctx) {
    NNOPS_ASSERT(src.data_type() == dst.data_type());
    switch (src.data_type()) {
    case DataType::f32: cl_to_planar_impl<float>(src, dst, ctx); return;
    case DataType::f16: cl_to_planar_impl<half>(src, dst, ctx);  return;
    case DataType::s8:  cl_to_planar_impl<int8_t>(src, dst, ctx); return;
    case DataType::u8:  cl_to_planar_impl<uint8_t>(src, dst, ctx); return;
    default: NNOPS_ASSERT(!"convert_channel_last_to_planar: unsupported data type");
    }
}

size_t channel_last_storage_bytes(const TensorDesc& logical_desc, int64_t alignment) {
    TensorDesc cl = logical_desc;
    if (cl.rank == 4) { cl.layout = TensorLayout::NHWC; }
    else if (cl.rank == 5) { cl.layout = TensorLayout::NDHWC; }
    return cl.storage_bytes(alignment);
}

}  // namespace nnops
