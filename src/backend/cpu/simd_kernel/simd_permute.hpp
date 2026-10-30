#pragma once
/// @file simd_permute.hpp
/// @brief Tiled 2D transpose via v_transpose_8x8 — the core of the Permute
/// operator's last-two-dims swap (2D transpose, batched (B,M,K)->(B,K,M),
/// NCHW<->NHCW).
///
/// Full 8×8 tiles transpose in registers with the unpack/permute intrinsics;
/// partial right/bottom edge tiles fall back to a scalar element copy. The
/// caller owns the parallelism — this takes one matrix.

#include "nnops/detail/simd/simd.hpp"

#include <algorithm>

namespace nnops::kernel {

using namespace simd;

/// Transpose [M × K] at @p in_ptr (row stride @p in_ld) to [K × M] at
/// @p out_ptr (row stride @p out_ld).
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
