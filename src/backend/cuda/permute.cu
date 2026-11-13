/// @file permute.cu
/// @brief CUDA Permute — a CuTe layout walk, plus a tiled transpose for the
///        one permutation that has no contiguous run to walk.
///
/// The general case is `move_common.cuh`: output axis d reads input axis
/// perm[d], so the source layout is the input's stride array reordered and the
/// destination is compact. A row of the output is a run at one source stride,
/// which is why the walk stays cheap even when no axis lines up.
///
/// The last-two swap is the exception that matters — it is the transpose
/// MatMul weights arrive in — because then the innermost output axis has the
/// input's *row* stride, so every element of a row lands in a different 128 B
/// line. Those take a 32x32 shared-memory tile instead, one block per tile per
/// batch slice. The tile route needs both operands dense (it indexes slices as
/// `m*k`), and falls back to the walk when they are not.
///
/// f32 and f16 in, out at the same dtype.

#include "move_common.cuh"
#include "nnops/ops/permute.hpp"

namespace nnops::backend::cuda {
namespace {

// ============================================================
// Tiled transpose (last-two swap)
// ============================================================

constexpr int kTransposeTile = 32;
constexpr int kTransposeRows = 8;

template <class T>
__global__ void transpose_tile_kernel(const T* __restrict__ src,
                                      T* __restrict__ dst,
                                      int64_t m, int64_t k)
{
    __shared__ T tile[kTransposeTile][kTransposeTile + 1];

    const int tx = int(threadIdx.x);
    const int ty = int(threadIdx.y);
    const T* s = src + int64_t(blockIdx.z) * m * k;
    T* d = dst + int64_t(blockIdx.z) * m * k;
    const int64_t m0 = int64_t(blockIdx.y) * kTransposeTile;
    const int64_t k0 = int64_t(blockIdx.x) * kTransposeTile;

    for (int j = 0; j < kTransposeTile; j += kTransposeRows) {
        const int64_t r = m0 + ty + j;
        const int64_t c = k0 + tx;
        if (r < m && c < k) { tile[ty + j][tx] = s[r * k + c]; }
    }
    __syncthreads();

    for (int j = 0; j < kTransposeTile; j += kTransposeRows) {
        const int64_t r = k0 + ty + j;
        const int64_t c = m0 + tx;
        if (r < k && c < m) { d[r * m + c] = tile[tx][ty + j]; }
    }
}

/// The last-two-swap route, when both operands are dense and the tile grid
/// fits CUDA's limits. Returns false to mean "use the layout walk".
template <class T>
bool try_transpose(const TensorView& input, TensorView& output,
                   cudaStream_t stream)
{
    const int64_t rank = input.rank();
    if (rank < 2) { return false; }
    if (!view_is_compact(input) || !view_is_compact(output)) { return false; }

    const int64_t m = input.shape(rank - 2);
    const int64_t k = input.shape(rank - 1);
    if (m == 0 || k == 0) { return true; }

    int64_t batch = 1;
    for (int64_t d = 0; d < rank - 2; ++d) { batch *= input.shape(d); }
    if (batch > 65535) { return false; }

    const dim3 grid(static_cast<unsigned>(ceil_div(k, int64_t(kTransposeTile))),
                    static_cast<unsigned>(ceil_div(m, int64_t(kTransposeTile))),
                    static_cast<unsigned>(batch));
    const dim3 block(kTransposeTile, kTransposeRows);
    transpose_tile_kernel<T><<<grid, block, 0, stream>>>(
        input.ptr<T>(), output.ptr<T>(), m, k);
    return true;
}

bool is_last_two_swap(const PermuteAttributes& attrs, int64_t rank)
{
    if (rank < 2) { return false; }
    for (int64_t i = 0; i < rank - 2; ++i) {
        if (attrs.perm[static_cast<size_t>(i)] != i) { return false; }
    }
    return attrs.perm[static_cast<size_t>(rank - 2)] == rank - 1 &&
           attrs.perm[static_cast<size_t>(rank - 1)] == rank - 2;
}

// ============================================================
// Entry
// ============================================================

template <class T>
Status permute_entry(const PermuteAttributes& attrs, TensorView& output,
                     const TensorView& input, cudaStream_t stream)
{
    const int64_t rank = input.rank();
    if (rank == 1) {
        const size_t bytes = static_cast<size_t>(input.numel()) * sizeof(T);
        if (bytes != 0) {
            CUDA_CHECK(cudaMemcpyAsync(output.ptr<T>(), input.ptr<T>(), bytes,
                                       cudaMemcpyDeviceToDevice, stream));
        }
        return Status::ok;
    }

    if (is_last_two_swap(attrs, rank) && try_transpose<T>(input, output, stream)) {
        return Status::ok;
    }

    int64_t dims[8], src_strides[8], dst_strides[8];
    fill_output_dims(output, dims);
    const int64_t total = output.numel();
    if (total == 0) { return Status::ok; }

    for (int64_t d = 0; d < rank; ++d) {
        src_strides[d] = input.stride_elems(attrs.perm[static_cast<size_t>(d)]);
        dst_strides[d] = output.stride_elems(d);
    }
    for (int64_t d = rank; d < 8; ++d) {
        src_strides[d] = 0;
        dst_strides[d] = 0;
    }

    MovePlan p;
    p.rank = rank;
    p.src = make_move_layout(dims, src_strides);
    p.dst = make_move_layout(dims, dst_strides);
    p.cols = dims[rank - 1];
    p.rows = total / p.cols;
    p.src_step = src_strides[rank - 1];
    p.dst_step = dst_strides[rank - 1];
    p.dst_compact = view_is_compact(output);
    for (int d = 0; d < 8; ++d) {
        p.dims[d] = dims[d];
        p.src_strides[d] = src_strides[d];
    }
    return launch_move(input.ptr<T>(), output.ptr<T>(), p, stream);
}

}  // anonymous namespace

Status permute_cuda(const PermuteAttributes& attrs, TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx, void* /*workspace*/)
{
    const auto stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return permute_entry<float>(attrs, output, inputs[0], stream);
    case DataType::f16:
        return permute_entry<__half>(attrs, output, inputs[0], stream);
    default:
        return Status::unsupported;
    }
}

}  // namespace nnops::backend::cuda
