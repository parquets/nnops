/// @file matmul_splitk.cuh
/// @brief The k-split path: cutting the reduction when the grid has no other
///        axis left to fill, and the scratch plane it lands in.
///
/// A grid shallower than the tile wants — the decode-shaped products, where M
/// fits one tile and N is a few dozen — leaves most of the machine idle,
/// because there is no third axis left to cut. The launcher then cuts the
/// reduction instead: blockIdx.z carries batch·k_splits + split, each block owns
/// a proportional range of the k tiles, and it writes its f32 accumulator tile
/// to a scratch plane instead of running the epilogue. A second kernel sums the
/// planes and applies the epilogue once, so alpha, bias, beta·C, the clamp, the
/// activation and the residual keep their single-pass order. Only the narrow
/// tiles split — the wide-tile gates already require a grid that fills the
/// machine — and the scratch is stream-ordered from a private, retained memory
/// pool (the default pool releases on every free, which measured at 1.45 ms a
/// pair on WDDM). `detail::cuda_splitk` forces or disables it for a bench; -1
/// means the heuristic.

#pragma once

#include "gemm_common.cuh"
#include "matmul_tile_h.cuh"
#include "matmul_tile_f.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace nnops::backend::cuda {

// ============================================================
// split-K reduction
// ============================================================

/// Sum the k-split planes and run the epilogue once, exactly as the single-pass
/// kernels would have. One thread per output element, planes summed in a fixed
/// order, so a given split count is deterministic. The block tile is not
/// involved: this pass is over the whole [M, N] plane.
template <class Elem> __global__ void dense_gemm_splitk_reduce_kernel(GemmParams p)
{
    const int64_t plane = static_cast<int64_t>(p.m) * p.n;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) {
        return;
    }

    const int m = int(idx / p.n);
    const int n = int(idx - static_cast<int64_t>(m) * p.n);
    const int batch = int(blockIdx.y);

    const float* src = p.partial + static_cast<int64_t>(batch) * p.k_splits * plane + idx;
    float sum = 0.0f;
    for (int s = 0; s < p.k_splits; ++s) {
        sum += src[static_cast<int64_t>(s) * plane];
    }

    Elem* C = static_cast<Elem*>(p.c) + detail::gemm_batch_offset(p, p.c_batch, batch);
    const int64_t off = static_cast<int64_t>(m) * p.ldc + n;

    float v = p.alpha * sum;
    if (p.bias != nullptr) {
        const Elem* bias = static_cast<const Elem*>(p.bias);
        v += gemm_to_float(bias[p.bias_row ? m : n]);
    }
    if (p.beta != 0.0f) {
        v += p.beta * gemm_to_float(C[off]);
    }
    v = fminf(fmaxf(v, p.min_clip), p.max_clip);
    v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
    if (p.residual != nullptr) {
        const Elem* residual =
            static_cast<const Elem*>(p.residual) + detail::gemm_batch_offset(p, p.r_batch, batch);
        v += gemm_to_float(residual[static_cast<int64_t>(m) * p.ldr + n]);
    }
    C[off] = Elem(v);
}

namespace detail {

template <class Elem> inline void launch_splitk_reduce(const GemmParams& p, cudaStream_t stream)
{
    constexpr int kThreads = 256;
    const int64_t plane = static_cast<int64_t>(p.m) * p.n;
    dim3 grid(static_cast<unsigned>((plane + kThreads - 1) / kThreads),
              static_cast<unsigned>(gemm_batch_count(p)), 1);
    dense_gemm_splitk_reduce_kernel<Elem><<<grid, kThreads, 0, stream>>>(p);
}

/// Launch a tile, cutting the reduction into k ranges when the grid is shallower
/// than the tile wants, and reducing the planes afterwards. Falls back to the
/// single-pass launch whenever a split cannot help or its scratch cannot be
/// carved.
template <class Elem, class Tile, bool TransA, bool TransB, bool Vectorized, int Groups = 1>
inline void launch_dense_split(const GemmParams& p, cudaStream_t stream)
{
    constexpr bool Half = !std::is_same_v<Elem, float>;
    auto launch_one = [&](const GemmParams& q) {
        if constexpr (Half) {
            launch_dense_h<Tile, Elem, TransA, TransB, Vectorized, Groups>(q, stream);
        } else {
            launch_dense_f<Tile, TransA, TransB, Vectorized>(q, stream);
        }
    };

    // A grouped CTA is wider and takes more smem, so it holds fewer residents:
    // two 39 KB blocks against three 27 KB ones.
    constexpr int blocks_per_sm = [] {
        if constexpr (Half) {
            return Groups == 1 ? 3 : 2;
        } else {
            return Tile::blocks;
        }
    }();
    const int elem_size = Half ? 2 : 4;
    int splits = pick_split_k(p, Tile::bM, Tile::bN * Groups, Tile::bK,
                              device_sm_count() * blocks_per_sm, elem_size);
    if (splits <= 1) {
        launch_one(p);
        return;
    }

    const int64_t batch = gemm_batch_count(p);
    const int64_t plane = static_cast<int64_t>(p.m) * p.n;
    // gridDim.z is 16-bit, and the partial planes are f32 whatever the operands.
    while (splits > 1 && batch * splits > 65535) {
        splits /= 2;
    }
    int64_t bytes = int64_t(splits) * batch * plane * int64_t(sizeof(float));
    while (splits > 1 && bytes > kSplitKMaxBytes) {
        splits /= 2;
        bytes = int64_t(splits) * batch * plane * int64_t(sizeof(float));
    }
    if (splits <= 1) {
        launch_one(p);
        return;
    }

    const GemmWorkspace ws = gemm_workspace_alloc(bytes, stream);
    if (ws.ptr == nullptr) {
        launch_one(p);
        return;
    }

    GemmParams q = p;
    q.k_splits = splits;
    q.partial = static_cast<float*>(ws.ptr);
    launch_one(q);
    if constexpr (Half) {
        launch_splitk_reduce<Elem>(q, stream);
    } else {
        launch_splitk_reduce<float>(q, stream);
    }
    gemm_workspace_free(ws, stream);
}

} // namespace detail

} // namespace nnops::backend::cuda
