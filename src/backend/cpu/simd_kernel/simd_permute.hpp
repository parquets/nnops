#pragma once
/// @file simd_permute.hpp
/// @brief SIMD kernel for 2D tiled transpose via v_transpose_8x8.
///
/// Core primitive for the Permute operator: transposes an [M × K] matrix
/// to [K × M] using 8×8 in-register SIMD transpose (unpack/permute
/// intrinsics). Handles partial edge tiles with scalar fallback.
///
/// Used by all "last-two-dims swap" permute patterns:
///   2D transpose, batched matrix transpose (B,M,K)→(B,K,M),
///   NCHW↔NHCW, multi-head dim reordering, etc.

#include "nnops/detail/simd/simd.hpp"

#include <algorithm>

namespace nnops::kernel {

using namespace simd;

/// Tiled 8×8 SIMD transpose of a 2D matrix (single-threaded, pure SIMD).
///
/// Transposes [M × K] at @p in_ptr (row stride = @p in_ld) to
///             [K × M] at @p out_ptr (row stride = @p out_ld).
///
/// Tiles are processed in row-major order. Full 8×8 tiles use
/// v_transpose_8x8 for in-register transpose; partial tiles at the
/// right and bottom edges fall back to scalar element-wise copy.
///
/// The caller is responsible for parallelism (e.g. batching over
/// leading dimensions or tiling over row groups).
///
/// @tparam T  Element type (f32 or f16)
/// @param in_ptr   Pointer to input matrix
/// @param out_ptr  Pointer to output matrix
/// @param M        Number of rows in input (= columns in output)
/// @param K        Number of columns in input (= rows in output)
/// @param in_ld    Leading dimension (row stride in elements) of input
/// @param out_ld   Leading dimension (row stride in elements) of output
template <typename T>
inline void tiled_transpose_2d(const T* in_ptr, T* out_ptr,
                                int64_t M, int64_t K,
                                int64_t in_ld, int64_t out_ld)
{
    constexpr int64_t TILE = 8;  // SIMD lane width for both f32/f16
    const int64_t num_tile_rows = (M + TILE - 1) / TILE;

    for (int64_t ti = 0; ti < num_tile_rows; ++ti) {
        const int64_t i = ti * TILE;          // start row in input
        const int64_t tile_m = std::min(TILE, M - i);

        for (int64_t j = 0; j < K; j += TILE) {
            const int64_t tile_k = std::min(TILE, K - j);

            if (tile_m == TILE && tile_k == TILE) {
                // Full 8×8 tile: load 8 rows, transpose, store 8 rows
                auto r0 = v_load(&in_ptr[(i + 0) * in_ld + j]);
                auto r1 = v_load(&in_ptr[(i + 1) * in_ld + j]);
                auto r2 = v_load(&in_ptr[(i + 2) * in_ld + j]);
                auto r3 = v_load(&in_ptr[(i + 3) * in_ld + j]);
                auto r4 = v_load(&in_ptr[(i + 4) * in_ld + j]);
                auto r5 = v_load(&in_ptr[(i + 5) * in_ld + j]);
                auto r6 = v_load(&in_ptr[(i + 6) * in_ld + j]);
                auto r7 = v_load(&in_ptr[(i + 7) * in_ld + j]);

                v_transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

                v_store(&out_ptr[(j + 0) * out_ld + i], r0);
                v_store(&out_ptr[(j + 1) * out_ld + i], r1);
                v_store(&out_ptr[(j + 2) * out_ld + i], r2);
                v_store(&out_ptr[(j + 3) * out_ld + i], r3);
                v_store(&out_ptr[(j + 4) * out_ld + i], r4);
                v_store(&out_ptr[(j + 5) * out_ld + i], r5);
                v_store(&out_ptr[(j + 6) * out_ld + i], r6);
                v_store(&out_ptr[(j + 7) * out_ld + i], r7);
            } else {
                // Partial tile at edge: scalar copy
                for (int64_t mi = 0; mi < tile_m; ++mi) {
                    for (int64_t kj = 0; kj < tile_k; ++kj) {
                        out_ptr[(j + kj) * out_ld + (i + mi)] =
                            in_ptr[(i + mi) * in_ld + (j + kj)];
                    }
                }
            }
        }
    }
}

}  // namespace nnops::kernel
