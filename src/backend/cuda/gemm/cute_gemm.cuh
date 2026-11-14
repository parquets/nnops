/// @file cute_gemm.cuh
/// @brief Dense GEMM kernels built on CuTe — the core shared by the CUDA
///        MatMul, Linear and Conv2D routes.
///
/// Two engines under one contract: f16 runs the SM80 16x8x16 tensor-core atom
/// with f32 accumulators over a 64x64x64 CTA tile, f32 runs a UniversalFMA SIMT
/// tile (64x128x16 once the grid is wide enough to refill the machine, else
/// 64x64x16), and both stream their k-slices through a cp.async pipeline. The
/// f32 tile swizzles its panel rows rather than padding them, which keeps a
/// four-block residency in 24 KB instead of 30 KB.
/// The gmem->smem copy is a 16 B vector load per thread whenever every row pitch
/// is 16 B aligned and the tile is fully in bounds; the alternative kernel
/// variant fills shared memory scalar-wise with a bounds check, which is also
/// what the edge tiles of an aligned call fall back to. That split is why the
/// smem layout follows the gmem order of each operand: a 16 B cp.async must land
/// contiguous in shared memory, and CuTe's LDSM atoms are the ones paired with
/// each of the two layouts (k-major -> LDSM_N, operand-major -> LDSM_T).
///
/// The epilogue is the operator-level contract, applied per element once the k
/// loop is done: scale the product, add the bias, add beta times the old C,
/// clamp, activate, then add a residual when the caller asked for it after the
/// activation (Conv2D and Linear's non-identity case).
///
/// Batching is numpy-style broadcast over up to four leading dimensions, each
/// carried as a (size, stride) pair and with stride 0 meaning broadcast;
/// blockIdx.z walks the flattened batch and the offsets are resolved in-kernel.
///
/// Split-K. A grid shallower than the tile wants — the decode-shaped products,
/// where M fits one tile and N is a few dozen — leaves most of the machine idle,
/// because there is no third axis left to cut. The launcher then cuts the
/// reduction instead: blockIdx.z carries batch·k_splits + split, each block owns
/// a proportional range of the k tiles, and it writes its f32 accumulator tile
/// to a scratch plane instead of running the epilogue. A second kernel sums the
/// planes and applies the epilogue once, so alpha, bias, beta·C, the clamp, the
/// activation and the residual keep their single-pass order. Only the narrow
/// tiles split — the wide-tile gates already require a grid that fills the
/// machine — and the scratch is stream-ordered from a private, retained memory
/// pool (the default pool releases on every free, which measured at 1.45 ms a
/// pair on WDDM). NNOPS_CUDA_SPLITK forces or disables it for a bench; unset
/// means the heuristic.

#pragma once

/// The engines live in their own headers, in dependency order: the shared
/// contract, the two CuTe tile kernels, the k-split launcher built on both, and
/// the hand-written narrow-m pair. What stays here is the choice between them.
#include "gemm_common.cuh"
#include "gemm_tile_h.cuh"
#include "gemm_tile_f.cuh"
#include "gemm_splitk.cuh"
#include "gemm_narrow.cuh"

namespace nnops::backend::cuda {

namespace detail {

/// Per-(dtype, transpose, vectorized) dispatch table.
template <class F>
inline void dispatch_dense(const GemmParams& p, bool trans_a, bool trans_b,
                           bool vectorized, F&& fn)
{
    const int key = (trans_a ? 2 : 0) + (trans_b ? 1 : 0) + (vectorized ? 4 : 0);
    switch (key) {
    case 0: fn.template operator()<false, false, false>(); break;
    case 1: fn.template operator()<false, true, false>(); break;
    case 2: fn.template operator()<true, false, false>(); break;
    case 3: fn.template operator()<true, true, false>(); break;
    case 4: fn.template operator()<false, false, true>(); break;
    case 5: fn.template operator()<false, true, true>(); break;
    case 6: fn.template operator()<true, false, true>(); break;
    default: fn.template operator()<true, true, true>(); break;
    }
}

}  // namespace detail

/// Launch one f32 GEMM with an explicit tile shape, bypassing the narrow/wide
/// heuristic. Every distinct tile (and every transpose/vectorized flag) is its
/// own kernel, so this is the benches' entry point for a tile search rather than
/// something the operators call.
template <class Tile, bool TransA, bool TransB, bool Vectorized = true>
inline void dense_gemm_f32_tile(const GemmParams& p, cudaStream_t stream)
{
    detail::launch_dense_f<Tile, TransA, TransB, Vectorized>(p, stream);
}

/// The f16 twin of `dense_gemm_f32_tile`: one tensor-core tile, spelled out.
template <class Tile, bool TransA, bool TransB, bool Vectorized = true>
inline void dense_gemm_f16_tile(const GemmParams& p, cudaStream_t stream)
{
    detail::launch_dense_h<Tile, cute::half_t, TransA, TransB, Vectorized>(p, stream);
}

/// The same tile in `Groups` copies side by side along n — one CTA of
/// `4·Groups` warps, which is how the bench prices warp residency without the
/// ldmatrix helpers having to accept a non-2x2 warp grid.
template <class Tile, int Groups, bool TransA, bool TransB, bool Vectorized = true>
inline void dense_gemm_f16_groups(const GemmParams& p, cudaStream_t stream)
{
    detail::launch_dense_h<Tile, cute::half_t, TransA, TransB, Vectorized, Groups>(p,
                                                                                  stream);
}

/// Dense C = alpha*A*B + beta*C + bias, epilogue applied, optional residual.
/// Returns false when the request has no kernel (bad extents).
inline bool dense_gemm_launch(const GemmParams& p, GemmElem elem,
                              bool trans_a, bool trans_b, bool vectorized,
                              cudaStream_t stream)
{
    if (p.m <= 0 || p.n <= 0 || p.k <= 0) { return false; }
    // Below a full tile of rows the mma is waste and B is the only operand that
    // matters, so the narrow-m loop takes the product instead of a tile.
    if (detail::small_m_applies(p, elem, trans_a, trans_b, vectorized)) {
        const bool done = elem == GemmElem::F16
                              ? detail::launch_small_m<cute::half_t>(p, stream)
                              : detail::launch_small_m<float>(p, stream);
        if (done) { return true; }
    }
    // The same rows, with the operand stored the other way round.
    if (detail::small_nt_applies(p, elem, trans_a, trans_b, vectorized)) {
        const bool done = elem == GemmElem::F16
                              ? detail::launch_small_nt<cute::half_t>(p, stream)
                              : detail::launch_small_nt<float>(p, stream);
        if (done) { return true; }
    }
    // The 128-wide tile reads A half as often, which takes a quarter off the
    // operand traffic, but its 26 KB of smem costs a resident block, so it only
    // pays once the grid can refill the machine several times over.
    const int sm = detail::device_sm_count();
    const long long wide_blocks = static_cast<long long>((p.m + 63) / 64) *
                                  ((p.n + 127) / 128) * detail::gemm_batch_count(p);
    const bool wide = sm > 0 && wide_blocks >= 4LL * sm;
    // f16/bf16 pick between one 64-wide tile a block and two of them: the paired
    // CTA shares the A panel between its groups, so it reads a quarter less, and
    // its eight warps put two on every SM sub-partition, where the four-warp
    // block exposes each ldmatrix and mma dependency. It is never slower than the
    // 128-wide tile it replaces -- 3.98 against 4.14 ms at 4096^3, 0.530 against
    // 0.563 at 512x4096x4096 -- so the wide tile is no longer dispatched.
    //
    // What it costs is block count, and below one wave that is not paid back by
    // the wider block: 128x128x4096 halves two blocks into one (-27%) and
    // 256x1024x1024 goes from 64 to 32 (-12%). One m tile is the exception and
    // the biggest win of all -- a single A panel shared across the whole row,
    // +39% to +63% at 16x4096x1024 through 32x2048x4096 -- so that case is taken
    // on its own. The k floor keeps it off the short-k shapes, where the extra
    // warps have nothing to hide behind.
    const int num_m = (p.m + 63) / 64;
    const long long pair_blocks = static_cast<long long>(num_m) *
                                  (((p.n + 63) / 64 + 1) / 2) *
                                  detail::gemm_batch_count(p);
    const bool pair_h =
        sm > 0 && p.k >= 1024 && (num_m == 1 || pair_blocks >= sm);
    detail::dispatch_dense(
        p, trans_a, trans_b, vectorized,
        [&]<bool TA, bool TB, bool V>() {
            if (elem == GemmElem::F16) {
                if (pair_h) {
                    detail::launch_dense_split<cute::half_t, kF16Narrow, TA, TB, V, 2>(
                        p, stream);
                } else {
                    detail::launch_dense_split<cute::half_t, kF16Narrow, TA, TB, V>(p,
                                                                                    stream);
                }
            } else if (elem == GemmElem::BF16) {
                if (pair_h) {
                    detail::launch_dense_split<cute::bfloat16_t, kF16Narrow, TA, TB, V,
                                               2>(p, stream);
                } else {
                    detail::launch_dense_split<cute::bfloat16_t, kF16Narrow, TA, TB, V>(
                        p, stream);
                }
            } else if (wide) {
                detail::launch_dense_f<kF32Wide, TA, TB, V>(p, stream);
            } else {
                detail::launch_dense_split<float, kF32Narrow, TA, TB, V>(p, stream);
            }
        });
    return true;
}

}  // namespace nnops::backend::cuda
