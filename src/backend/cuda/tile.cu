/// @file tile.cu
/// @brief CUDA Tile and Expand — the two ways to repeat a tensor.
///
/// Expand is a plain broadcast: an input axis of extent 1 (or one the right-
/// aligned target does not reach) has source stride 0, so it is the shared
/// layout walk with a zero stride and needs nothing of its own.
///
/// Tile repeats, and a repeat is `o % n`, which no layout expresses. The output
/// is instead cut into periods: the last axis's `repeats[rank-1]` copies of the
/// input row are written one after another from one source row, and the outer
/// axes are decoded once per row with the modulo applied there. That is the same
/// cost as the plain walk — one decode per row — and the inner run keeps its
/// vectorized copy.
///
/// f32, f16 and the integer dtypes travel as bytes: a tile never touches the
/// values.

#include "move_common.cuh"
#include "nnops/ops/tile.hpp"

namespace nnops::backend::cuda {
namespace {

// ============================================================
// Expand — a broadcast layout
// ============================================================

template <class T>
Status expand_entry(const TensorView& input, TensorView& output,
                    cudaStream_t stream)
{
    const int64_t orank = output.rank();
    const int64_t irank = input.rank();
    const int64_t total = output.numel();
    if (total == 0) { return Status::ok; }

    int64_t dims[8], src_strides[8], dst_strides[8];
    fill_output_dims(output, dims);
    fill_view_strides(output, dst_strides);
    for (int64_t d = 0; d < 8; ++d) { src_strides[d] = 0; }
    for (int64_t d = 0; d < orank; ++d) {
        const int64_t ii = irank - orank + d;
        if (ii >= 0 && input.shape(ii) != 1) {
            src_strides[d] = input.stride_elems(ii);
        }
    }

    MovePlan p;
    p.rank = orank;
    p.src = make_move_layout(dims, src_strides);
    p.dst = make_move_layout(dims, dst_strides);
    p.cols = dims[orank - 1];
    p.rows = total / p.cols;
    p.src_step = src_strides[orank - 1];
    p.dst_step = dst_strides[orank - 1];
    p.dst_compact = view_is_compact(output);
    for (int d = 0; d < 8; ++d) {
        p.dims[d] = dims[d];
        p.src_strides[d] = src_strides[d];
    }
    return launch_move(input.ptr<T>(), output.ptr<T>(), p, stream);
}

// ============================================================
// Tile — one row per outer coordinate, repeated along the last axis
// ============================================================

struct TileGeom {
    int64_t rows = 0;          ///< outer coordinates (axes 0..rank-2)
    int64_t cols = 0;          ///< input extent of the last axis
    int64_t rep_last = 1;      ///< repeats of the last axis
    int64_t out_last = 0;      ///< output extent of the last axis
    int64_t rank = 0;
    int64_t outer[8] = {};     ///< output extents of axes 0..rank-2
    int64_t in_dims[8] = {};
    int64_t in_strides[8] = {};
    int64_t in_step = 1;       ///< input stride along the last axis
    int64_t out_step = 1;      ///< output stride along the last axis
    MoveLayout dst;
    bool dst_compact = true;
};

template <class T>
__global__ void tile_rows_kernel(const T* __restrict__ src,
                                 T* __restrict__ dst, TileGeom g)
{
    for (int64_t q = blockIdx.x; q < g.rows; q += gridDim.x) {
        int64_t rem = q;
        int64_t so = 0;
        for (int64_t d = g.rank - 2; d >= 0; --d) {
            const int64_t c = (g.outer[d] > 0) ? rem % g.outer[d] : 0;
            rem = (g.outer[d] > 0) ? rem / g.outer[d] : 0;
            const int64_t n = g.in_dims[d];
            so += (n > 0 ? c % n : 0) * g.in_strides[d];
        }

        const int64_t flat = q * g.out_last;
        const int64_t base = g.dst_compact ? flat : g.dst(flat);
        for (int64_t j = 0; j < g.rep_last; ++j) {
            copy_run(dst + base + j * g.cols * g.out_step, src + so, g.cols,
                     g.in_step, g.out_step);
        }
    }
}

template <class T>
Status tile_entry(TensorView& output, const TensorView& input,
                  cudaStream_t stream)
{
    const int64_t rank = input.rank();
    const int64_t total = output.numel();
    if (total == 0) { return Status::ok; }

    int64_t dims[8], dst_strides[8];
    fill_output_dims(output, dims);
    fill_view_strides(output, dst_strides);

    TileGeom g;
    g.rank = rank;
    g.cols = input.shape(rank - 1);
    g.out_last = dims[rank - 1];
    g.rep_last = (g.cols > 0) ? g.out_last / g.cols : 1;
    for (int64_t d = 0; d < rank - 1; ++d) { g.outer[d] = dims[d]; }
    fill_view_strides(input, g.in_strides);
    for (int64_t d = 0; d < rank; ++d) { g.in_dims[d] = input.shape(d); }
    g.rows = total / g.out_last;
    g.in_step = g.in_strides[rank - 1];
    g.out_step = dst_strides[rank - 1];
    g.dst = make_move_layout(dims, dst_strides);
    g.dst_compact = view_is_compact(output);

    const unsigned grid =
        static_cast<unsigned>(std::min<int64_t>(g.rows, int64_t(1) << 20));
    tile_rows_kernel<T><<<grid, kCudaBlockSize, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(), g);
    return Status::ok;
}

}  // anonymous namespace

Status tile_cuda(const TileAttributes& /*attrs*/, TensorView& output,
                 std::span<const TensorView> inputs,
                 const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:  return tile_entry<float>(output, inputs[0], stream);
    case DataType::f16:  return tile_entry<__half>(output, inputs[0], stream);
    case DataType::bf16: return tile_entry<__nv_bfloat16>(output, inputs[0], stream);
    case DataType::s8:   return tile_entry<int8_t>(output, inputs[0], stream);
    case DataType::u8:   return tile_entry<uint8_t>(output, inputs[0], stream);
    case DataType::s32:  return tile_entry<int32_t>(output, inputs[0], stream);
    case DataType::s64:  return tile_entry<int64_t>(output, inputs[0], stream);
    default:             return Status::unsupported;
    }
}

Status expand_cuda(const ExpandAttributes& /*attrs*/, TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:  return expand_entry<float>(inputs[0], output, stream);
    case DataType::f16:  return expand_entry<__half>(inputs[0], output, stream);
    case DataType::bf16: return expand_entry<__nv_bfloat16>(inputs[0], output, stream);
    case DataType::s8:   return expand_entry<int8_t>(inputs[0], output, stream);
    case DataType::u8:   return expand_entry<uint8_t>(inputs[0], output, stream);
    case DataType::s32:  return expand_entry<int32_t>(inputs[0], output, stream);
    case DataType::s64:  return expand_entry<int64_t>(inputs[0], output, stream);
    default:             return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
