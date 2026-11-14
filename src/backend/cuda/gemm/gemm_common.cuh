/// @file gemm_common.cuh
/// @brief The dense GEMM kernel contract: the parameter struct every kernel
///        takes, the shared epilogue, and the shared-memory layouts and block
///        geometry the tile engines are built from.
///
/// The epilogue is the operator-level contract, applied per element once the k
/// loop is done: scale the product, add the bias, add beta times the old C,
/// clamp, activate, then add a residual when the caller asked for it after the
/// activation (Conv2D and Linear's non-identity case).
///
/// Batching is numpy-style broadcast over up to four leading dimensions, each
/// carried as a (size, stride) pair and with stride 0 meaning broadcast;
/// blockIdx.z walks the flattened batch and the offsets are resolved in-kernel.

#pragma once

#include <cute/tensor.hpp>
#include <nnops/core/epilogue.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace nnops::backend::cuda {


/// The dense GEMM's element formats: the f32 SIMT tile, or one of the two
/// 2-byte tensor-core formats (same MMA shape, different operand format).
enum class GemmElem { F32, F16, BF16 };

/// Everything a dense GEMM needs besides the element type and the two
/// transpose flags (which are template parameters, since they pick the copy
/// atoms). Strides are in elements; batch strides are 0 for a broadcast axis.
struct GemmParams {
    const void* a = nullptr;
    const void* b = nullptr;
    void* c = nullptr;
    const void* bias = nullptr;
    const void* residual = nullptr;

    int m = 0, n = 0, k = 0;
    int64_t lda = 0, ldb = 0, ldc = 0, ldr = 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    EpilogueActivateType epilogue = EpilogueActivateType::None;
    float ep_alpha = 0.0f;
    float ep_beta = 1.0f;
    float min_clip = -std::numeric_limits<float>::infinity();
    float max_clip = std::numeric_limits<float>::infinity();

    bool bias_row = false;

    // ---- split-K ----------------------------------------------------------
    // `k_splits` cuts the reduction into that many ranges, one per blockIdx.z
    // slice (z = batch · k_splits + split); each block writes its f32
    // accumulator tile to `partial` instead of running the epilogue, and a
    // second kernel sums the planes and applies the epilogue once. 1 means the
    // plain single-pass kernel and `partial` unused.
    int k_splits = 1;
    float* partial = nullptr;  ///< [batch][split][m][n], row-major

    int batch_rank = 0;
    int64_t batch_size[4] = {1, 1, 1, 1};
    int64_t a_batch[4] = {0, 0, 0, 0};
    int64_t b_batch[4] = {0, 0, 0, 0};
    int64_t c_batch[4] = {0, 0, 0, 0};
    int64_t r_batch[4] = {0, 0, 0, 0};
};

// ============================================================
// Epilogue
// ============================================================

__device__ inline float gemm_apply_epilogue(
    EpilogueActivateType ep, float alpha, float beta, float x)
{
    switch (ep) {
    case EpilogueActivateType::None:
        return x;
    case EpilogueActivateType::Relu:
        return fmaxf(x, 0.0f);
    case EpilogueActivateType::Relu6:
        return fminf(fmaxf(x, 0.0f), 6.0f);
    case EpilogueActivateType::LeakyRelu:
        return (x > 0.0f) ? x : alpha * x;
    case EpilogueActivateType::Sigmoid:
        return 1.0f / (1.0f + expf(-x));
    case EpilogueActivateType::Tanh:
        return tanhf(x);
    case EpilogueActivateType::Gelu: {
        constexpr float c = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
    }
    case EpilogueActivateType::Silu:
        return x / (1.0f + expf(-x));
    case EpilogueActivateType::HardSwish: {
        const float r = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        return x * r * (beta / 6.0f);
    }
    case EpilogueActivateType::Elu:
        return (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
    return x;
}

template <class Elem>
__device__ inline float gemm_to_float(Elem x) { return float(x); }

template <>
__device__ inline float gemm_to_float<float>(float x) { return x; }

// ============================================================
// Shapes
// ============================================================

namespace detail {

constexpr int kGemmThreads = 128;

/// Shared memory a block can ask for on this class of part (GA10x: 128 KB minus
/// the L1 carveout). A tile's `blocks` target has to fit under it.
constexpr int kSmemPerSm = 100 * 1024;

inline constexpr auto kGemmAtomK = cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
                 cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{});

inline constexpr auto kGemmAtomMN = cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_64, cute::_8>,
                 cute::Stride<cute::_1, cute::_64>>{});

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
                    cute::make_stride(cute::Int<int(BK{}) + kGemmPadK>{},
                                      cute::_1{})),
                cute::make_shape(bm, bk, p));
        } else {
            return cute::tile_to_shape(kGemmAtomK, cute::make_shape(bm, bk, p));
        }
    } else {
        return cute::tile_to_shape(kGemmAtomMN, cute::make_shape(bm, bk, p));
    }
}

/// The batch index a block owns. Without split-K it is blockIdx.z; with it,
/// blockIdx.z carries batch · k_splits + split, split varying fastest so the
/// slices of one matrix land on different waves of the scheduler.
__device__ inline int gemm_block_batch(const GemmParams& p)
{
    const int z = int(blockIdx.z);
    return p.k_splits > 1 ? z / p.k_splits : z;
}

__device__ inline int gemm_block_split(const GemmParams& p)
{
    return p.k_splits > 1 ? int(blockIdx.z) % p.k_splits : 0;
}

__device__ inline int64_t gemm_batch_offset(const GemmParams& p,
                                            const int64_t* strides,
                                            int64_t idx)
{
    int64_t off = 0;
    for (int d = 0; d < p.batch_rank; ++d) {
        const int64_t i = idx % p.batch_size[d];
        idx /= p.batch_size[d];
        off += i * strides[d];
    }
    return off;
}

/// The k-tile range a split owns, as [begin, begin + count). The bounds are
/// proportional cuts of the tile count, so with `k_splits <= k_tiles` every
/// split is non-empty and the ranges differ by at most one tile.
__device__ inline void gemm_split_range(const GemmParams& p, int bK, int split,
                                        int& kt_begin, int& kt_count)
{
    const int k_tiles = (p.k + bK - 1) / bK;
    if (p.k_splits <= 1) {
        kt_begin = 0;
        kt_count = k_tiles;
        return;
    }
    const int64_t tiles = k_tiles;
    kt_begin = int(tiles * split / p.k_splits);
    kt_count = int(tiles * (split + 1) / p.k_splits) - kt_begin;
}

}  // namespace detail

namespace detail {

inline int gemm_batch_count(const GemmParams& p)
{
    int count = 1;
    for (int d = 0; d < p.batch_rank; ++d) { count *= int(p.batch_size[d]); }
    return count;
}

/// Queried once per process; 0 when unavailable, which keeps the narrow tile.
inline int device_sm_count()
{
    static const int count = [] {
        int dev = 0;
        int n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { return 0; }
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            return 0;
        }
        return n;
    }();
    return count;
}

/// Queried once per process; 0 when unavailable, which drops the plane cap.
inline int64_t device_l2_bytes()
{
    static const int64_t bytes = [] {
        int dev = 0;
        int n = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) { return int64_t{0}; }
        if (cudaDeviceGetAttribute(&n, cudaDevAttrL2CacheSize, dev) != cudaSuccess) {
            return int64_t{0};
        }
        return int64_t(n);
    }();
    return bytes;
}

}  // namespace detail

}  // namespace nnops::backend::cuda
