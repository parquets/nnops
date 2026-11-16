/// @file matmul_common.cuh
/// @brief The tile machinery the dense GEMM engines are built from: the
///        shared-memory layouts, their swizzle atoms, and the block thread
///        count. The parameter struct, the epilogue contract and the split-K
///        scratch are CuTe-free and live in gemm_common.cuh.

#pragma once

#include "gemm_common.cuh"

#include <cute/tensor.hpp>

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {

/// The dense GEMM's element formats: the f32 SIMT tile, or one of the two
/// 2-byte tensor-core formats (same MMA shape, different operand format).
enum class GemmElem {
    F32,
    F16,
    BF16
};

// ============================================================
// Shapes
// ============================================================

namespace detail {

inline constexpr auto kGemmAtomK =
    cute::composition(cute::Swizzle<3, 3, 3>{},
                      cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
                                   cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{});

inline constexpr auto kGemmAtomMN = cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_64, cute::_8>, cute::Stride<cute::_1, cute::_64>>{});

/// A k tile narrower than the swizzle atom's 64 halves cannot use the XOR
/// layout -- the atom would not divide it. An 8-half pad serves the same end by
/// a different route: it walks consecutive rows 80 B apart, so the eight rows an
/// LDSM reads land on eight different bank groups.
constexpr int kGemmPadK = 8;

template <bool KMajor, class BM, class BK, class P>
constexpr auto gemm_smem_layout(BM bm, BK bk, P p)
{
    if constexpr (KMajor) {
        if constexpr (int(BK{}) % 64 != 0) {
            return cute::tile_to_shape(
                cute::make_layout(
                    cute::make_shape(bm, bk),
                    cute::make_stride(cute::Int<int(BK{}) + kGemmPadK>{}, cute::_1{})),
                cute::make_shape(bm, bk, p));
        } else {
            return cute::tile_to_shape(kGemmAtomK, cute::make_shape(bm, bk, p));
        }
    } else {
        return cute::tile_to_shape(kGemmAtomMN, cute::make_shape(bm, bk, p));
    }
}

} // namespace detail

} // namespace nnops::backend::cuda
