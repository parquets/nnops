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

#pragma once

#include <cute/tensor.hpp>
#include <nnops/core/epilogue.hpp>

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>

namespace nnops::backend::cuda {

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

inline constexpr auto kGemmAtomK = cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_8, cute::Shape<cute::_8, cute::_8>>,
                 cute::Stride<cute::_8, cute::Stride<cute::_1, cute::_64>>>{});

inline constexpr auto kGemmAtomMN = cute::composition(
    cute::Swizzle<3, 3, 3>{},
    cute::Layout<cute::Shape<cute::_64, cute::_8>,
                 cute::Stride<cute::_1, cute::_64>>{});

template <bool KMajor, class BM, class BK, class P>
constexpr auto gemm_smem_layout(BM bm, BK bk, P p)
{
    if constexpr (KMajor) {
        return cute::tile_to_shape(kGemmAtomK, cute::make_shape(bm, bk, p));
    } else {
        return cute::tile_to_shape(kGemmAtomMN, cute::make_shape(bm, bk, p));
    }
}

__device__ inline int64_t gemm_batch_offset(const GemmParams& p,
                                            const int64_t* strides)
{
    int64_t idx = blockIdx.z;
    int64_t off = 0;
    for (int d = 0; d < p.batch_rank; ++d) {
        const int64_t i = idx % p.batch_size[d];
        idx /= p.batch_size[d];
        off += i * strides[d];
    }
    return off;
}

}  // namespace detail

// ============================================================
// f16 tensor-core kernel
// ============================================================

template <bool TransA, bool TransB, bool Vectorized>
__global__ __launch_bounds__(detail::kGemmThreads)
void dense_gemm_h_kernel(GemmParams p)
{
    using namespace cute;
    using In = cute::half_t;
    using Out = cute::half_t;

    constexpr int bM = 64;
    constexpr int bN = 64;
    constexpr int bK = 64;
    constexpr int kStages = 3;

    const int M = p.m;
    const int N = p.n;
    const int K = p.k;
    const int m0 = int(blockIdx.x) * bM;
    const int n0 = int(blockIdx.y) * bN;

    In const* A = static_cast<In const*>(p.a) + detail::gemm_batch_offset(p, p.a_batch);
    In const* B = static_cast<In const*>(p.b) + detail::gemm_batch_offset(p, p.b_batch);
    In const* bias = static_cast<In const*>(p.bias);
    Out* C = static_cast<Out*>(p.c) + detail::gemm_batch_offset(p, p.c_batch);
    Out const* residual = static_cast<Out const*>(p.residual) +
                          detail::gemm_batch_offset(p, p.r_batch);

    Tensor mA = [&] {
        if constexpr (TransA) {
            return make_tensor(make_gmem_ptr(A), make_shape(M, K),
                               make_stride(Int<1>{}, int(p.lda)));
        } else {
            return make_tensor(make_gmem_ptr(A), make_shape(M, K),
                               make_stride(int(p.lda), Int<1>{}));
        }
    }();
    Tensor mB = [&] {
        if constexpr (TransB) {
            return make_tensor(make_gmem_ptr(B), make_shape(N, K),
                               make_stride(int(p.ldb), Int<1>{}));
        } else {
            return make_tensor(make_gmem_ptr(B), make_shape(N, K),
                               make_stride(Int<1>{}, int(p.ldb)));
        }
    }();
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N),
                            make_stride(int(p.ldc), Int<1>{}));

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    constexpr auto sA_layout = detail::gemm_smem_layout<!TransA>(
        Int<bM>{}, Int<bK>{}, Int<kStages>{});
    constexpr auto sB_layout = detail::gemm_smem_layout<TransB>(
        Int<bN>{}, Int<bK>{}, Int<kStages>{});

    constexpr int kSmemA = int(cosize(sA_layout));
    constexpr int kSmemB = int(cosize(sB_layout));

    extern __shared__ char gemm_smem[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<In*>(gemm_smem)), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<In*>(gemm_smem + kSmemA * 2)),
                            sB_layout);

    TiledMMA mma = make_tiled_mma(SM80_16x8x16_F32F16F16F32_TN{},
                                  Layout<Shape<_2, _2, _1>>{},
                                  Tile<_32, _32, _16>{});

    auto copy_a = [&] {
        if constexpr (TransA) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, In>{},
                Layout<Shape<_8, _16>>{},
                Layout<Shape<_8, _1>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, In>{},
                Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                Layout<Shape<_1, _8>>{});
        }
    }();
    auto copy_b = [&] {
        if constexpr (TransB) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, In>{},
                Layout<Shape<_16, _8>, Stride<_8, _1>>{},
                Layout<Shape<_1, _8>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, In>{},
                Layout<Shape<_8, _16>>{},
                Layout<Shape<_8, _1>>{});
        }
    }();

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    const bool m_full = (m0 + bM <= M);
    const bool n_full = (n0 + bN <= N);

    auto fill_a = [&](int kt, int stage) {
        const int k0 = kt * bK;
        for (int i = threadIdx.x; i < bM * bK; i += blockDim.x) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int m = m0 + row;
            const int k = k0 + col;
            sA(row, col, stage) =
                (m < M && k < K) ? gA(row, col, kt) : In(0);
        }
    };
    auto fill_b = [&](int kt, int stage) {
        const int k0 = kt * bK;
        for (int i = threadIdx.x; i < bN * bK; i += blockDim.x) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int n = n0 + row;
            const int k = k0 + col;
            sB(row, col, stage) =
                (n < N && k < K) ? gB(row, col, kt) : In(0);
        }
    };
    auto load_a = [&](int kt, int stage) {
        if constexpr (Vectorized) {
            if (m_full && (kt + 1) * bK <= K) {
                copy(copy_a, tAgA(_, _, _, kt), tAsA(_, _, _, stage));
                return;
            }
        }
        fill_a(kt, stage);
    };
    auto load_b = [&](int kt, int stage) {
        if constexpr (Vectorized) {
            if (n_full && (kt + 1) * bK <= K) {
                copy(copy_b, tBgB(_, _, _, kt), tBsB(_, _, _, stage));
                return;
            }
        }
        fill_b(kt, stage);
    };

    int k_tile_count = (K + bK - 1) / bK;
    int k_tile_next = 0;

    CUTE_UNROLL
    for (int pipe = 0; pipe < kStages - 1; ++pipe) {
        load_a(k_tile_next, pipe);
        load_b(k_tile_next, pipe);
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    ThrMMA thr_mma = mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, 0));
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);

    auto s2r_atom_a = [&] {
        if constexpr (TransA) {
            return Copy_Atom<SM75_U16x8_LDSM_T, In>{};
        } else {
            return Copy_Atom<SM75_U32x4_LDSM_N, In>{};
        }
    }();
    auto s2r_atom_b = [&] {
        if constexpr (TransB) {
            return Copy_Atom<SM75_U32x4_LDSM_N, In>{};
        } else {
            return Copy_Atom<SM75_U16x8_LDSM_T, In>{};
        }
    }();

    TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
    Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
    Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

    int smem_pipe_read = 0;
    int smem_pipe_write = kStages - 1;
    Tensor tXsA_p = tXsA(_, _, _, smem_pipe_read);
    Tensor tXsB_p = tXsB(_, _, _, smem_pipe_read);

    auto K_BLOCK_MAX = size<2>(tCrA);

    if constexpr (decltype(K_BLOCK_MAX)::value > 1) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        copy(s2r_atom_a, tXsA_p(_, _, Int<0>{}), tXrA(_, _, Int<0>{}));
        copy(s2r_atom_b, tXsB_p(_, _, Int<0>{}), tXrB(_, _, Int<0>{}));
    }

    CUTE_NO_UNROLL
    while (k_tile_count > -(kStages - 1)) {
        CUTE_UNROLL
        for (int k_block = 0; k_block < decltype(K_BLOCK_MAX)::value; ++k_block) {
            if (k_block == decltype(K_BLOCK_MAX)::value - 1) {
                tXsA_p = tXsA(_, _, _, smem_pipe_read);
                tXsB_p = tXsB(_, _, _, smem_pipe_read);
                cp_async_wait<kStages - 2>();
                __syncthreads();
            }
            const int k_block_next =
                (k_block + 1) % decltype(K_BLOCK_MAX)::value;
            copy(s2r_atom_a, tXsA_p(_, _, k_block_next), tXrA(_, _, k_block_next));
            copy(s2r_atom_b, tXsB_p(_, _, k_block_next), tXrB(_, _, k_block_next));
            if (k_block == 0) {
                load_a(k_tile_next, smem_pipe_write);
                load_b(k_tile_next, smem_pipe_write);
                cp_async_fence();
                --k_tile_count;
                if (k_tile_count > 0) { ++k_tile_next; }
                smem_pipe_write = smem_pipe_read;
                smem_pipe_read =
                    (smem_pipe_read == kStages - 1) ? 0 : smem_pipe_read + 1;
            }
            gemm(mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
        }
    }

    // The coordinate tensor is block-tiled exactly like gC: partition_C only
    // applies the thread partition, so handing it the whole (M, N) would give
    // every block the coordinates of block (0, 0).
    Tensor cC = make_identity_tensor(make_shape(M, N));
    Tensor tCcC = thr_mma.partition_C(
        local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{}));

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }

        float v = p.alpha * float(tCrC(i));
        if (bias != nullptr) {
            v += gemm_to_float(bias[p.bias_row ? m : n]);
        }
        if (p.beta != 0.0f) {
            v += p.beta * float(tCgC(i));
        }
        v = fminf(fmaxf(v, p.min_clip), p.max_clip);
        v = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, v);
        if (residual != nullptr) {
            v += float(residual[m * p.ldr + n]);
        }
        tCgC(i) = Out(v);
    }
}

// ============================================================
// f32 SIMT kernel
// ============================================================

template <bool TransA, bool TransB, bool Vectorized, bool Wide>
__global__ __launch_bounds__(detail::kGemmThreads, 4)
void dense_gemm_f_kernel(GemmParams p)
{
    using namespace cute;
    using Elem = float;

    constexpr int bM = 64;
    constexpr int bN = Wide ? 128 : 64;
    constexpr int bK = 16;
    constexpr int kStages = Wide ? 2 : 3;

    const int M = p.m;
    const int N = p.n;
    const int K = p.k;
    const int m0 = int(blockIdx.x) * bM;
    const int n0 = int(blockIdx.y) * bN;

    Elem const* A = static_cast<Elem const*>(p.a) + detail::gemm_batch_offset(p, p.a_batch);
    Elem const* B = static_cast<Elem const*>(p.b) + detail::gemm_batch_offset(p, p.b_batch);
    Elem const* bias = static_cast<Elem const*>(p.bias);
    Elem* C = static_cast<Elem*>(p.c) + detail::gemm_batch_offset(p, p.c_batch);
    Elem const* residual = static_cast<Elem const*>(p.residual) +
                           detail::gemm_batch_offset(p, p.r_batch);

    Tensor mA = [&] {
        if constexpr (TransA) {
            return make_tensor(make_gmem_ptr(A), make_shape(M, K),
                               make_stride(Int<1>{}, int(p.lda)));
        } else {
            return make_tensor(make_gmem_ptr(A), make_shape(M, K),
                               make_stride(int(p.lda), Int<1>{}));
        }
    }();
    Tensor mB = [&] {
        if constexpr (TransB) {
            return make_tensor(make_gmem_ptr(B), make_shape(N, K),
                               make_stride(int(p.ldb), Int<1>{}));
        } else {
            return make_tensor(make_gmem_ptr(B), make_shape(N, K),
                               make_stride(Int<1>{}, int(p.ldb)));
        }
    }();
    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});

    // A k-contiguous panel is bK = 16 floats = 64 B wide, so each row can start
    // in bank 0 or bank 16 only: without help the eight rows a warp reads at one
    // k land on two banks. Padding the row is one fix (20 floats); the swizzle is
    // the other, XORing the row index into the 16 B vector index so the eight
    // rows get the same distinct phases at no shared-memory cost -- and bits 0..1
    // stay put, so the cp.async vector is still 16 B. A panel wide in the
    // operand direction (>= 64 floats, two or more full bank cycles per row) is
    // already conflict free and takes neither.
    constexpr auto swz = Swizzle<3, 2, 3>{};
    constexpr auto sA_layout = [&] {
        if constexpr (TransA) {
            return make_layout(make_shape(Int<bM>{}, Int<bK>{}, Int<kStages>{}),
                               make_stride(Int<1>{}, Int<bM>{},
                                           Int<bM * bK>{}));
        } else {
            return composition(
                swz, make_layout(make_shape(Int<bM>{}, Int<bK>{}, Int<kStages>{}),
                                 make_stride(Int<bK>{}, Int<1>{},
                                             Int<bK * bM>{})));
        }
    }();
    constexpr auto sB_layout = [&] {
        if constexpr (TransB) {
            return composition(
                swz, make_layout(make_shape(Int<bN>{}, Int<bK>{}, Int<kStages>{}),
                                 make_stride(Int<bK>{}, Int<1>{},
                                             Int<bK * bN>{})));
        } else {
            return make_layout(make_shape(Int<bN>{}, Int<bK>{}, Int<kStages>{}),
                               make_stride(Int<1>{}, Int<bN>{},
                                           Int<bN * bK>{}));
        }
    }();

    constexpr int kSmemA = int(cosize(sA_layout));
    constexpr int kSmemB = int(cosize(sB_layout));

    extern __shared__ char gemm_smem[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<Elem*>(gemm_smem)), sA_layout);
    Tensor sB = make_tensor(
        make_smem_ptr(reinterpret_cast<Elem*>(gemm_smem + kSmemA * sizeof(Elem))),
        sB_layout);

    // Thread tile: 8 rows each, and a *contiguous* strip of Tn columns. A CuTe
    // tiled-MMA hands every thread its N values cyclically (n = t_n + 16*j), and
    // those four values are 16 floats apart in an n-contiguous panel, so no
    // shared-memory orientation can serve them as one 16 B load -- each became a
    // scalar LDS and the loop carried 0.19 shared loads per FFMA where cuBLAS
    // carries 0.068. Mapping the strip by hand makes both operands 16 B loads:
    // A along its k-contiguous axis (or along m when the panel is m-contiguous),
    // B along n (or along k), for 0.094 / 0.063 shared loads per FFMA.
    constexpr int Tm = 8;
    constexpr int Tn = bN / 16;
    constexpr int kThreadsM = bM / Tm;
    static_assert(kThreadsM * (bN / Tn) == detail::kGemmThreads);

    const int tx = int(threadIdx.x) % kThreadsM;
    const int ty = int(threadIdx.x) / kThreadsM;
    auto arow = [&](int j) { return TransA ? tx * Tm + j : tx + kThreadsM * j; };
    auto brow = [&](int j) { return TransB ? ty + 16 * j : ty * Tn + j; };

    auto copy_a = [&] {
        if constexpr (TransA) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_16, _8>>{},
                Layout<Shape<_4, _1>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_32, _4>, Stride<_4, _1>>{},
                Layout<Shape<_1, _4>>{});
        }
    }();
    auto copy_b = [&] {
        if constexpr (TransB) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_32, _4>, Stride<_4, _1>>{},
                Layout<Shape<_1, _4>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<_16, _8>>{},
                Layout<Shape<_4, _1>>{});
        }
    }();

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);

    ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB);
    Tensor tBsB = thr_copy_b.partition_D(sB);

    const bool m_full = (m0 + bM <= M);
    const bool n_full = (n0 + bN <= N);

    auto fill_a = [&](int kt, int stage) {
        const int k0 = kt * bK;
        for (int i = threadIdx.x; i < bM * bK; i += blockDim.x) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int m = m0 + row;
            const int k = k0 + col;
            sA(row, col, stage) =
                (m < M && k < K) ? gA(row, col, kt) : Elem(0);
        }
    };
    auto fill_b = [&](int kt, int stage) {
        const int k0 = kt * bK;
        for (int i = threadIdx.x; i < bN * bK; i += blockDim.x) {
            const int row = i / bK;
            const int col = i - row * bK;
            const int n = n0 + row;
            const int k = k0 + col;
            sB(row, col, stage) =
                (n < N && k < K) ? gB(row, col, kt) : Elem(0);
        }
    };
    auto load_a = [&](int kt, int stage) {
        if constexpr (Vectorized) {
            if (m_full && (kt + 1) * bK <= K) {
                copy(copy_a, tAgA(_, _, _, kt), tAsA(_, _, _, stage));
                return;
            }
        }
        fill_a(kt, stage);
    };
    auto load_b = [&](int kt, int stage) {
        if constexpr (Vectorized) {
            if (n_full && (kt + 1) * bK <= K) {
                copy(copy_b, tBgB(_, _, _, kt), tBsB(_, _, _, stage));
                return;
            }
        }
        fill_b(kt, stage);
    };

    int k_tile_count = (K + bK - 1) / bK;
    int k_tile_next = 0;

    CUTE_UNROLL
    for (int pipe = 0; pipe < kStages - 1; ++pipe) {
        load_a(k_tile_next, pipe);
        load_b(k_tile_next, pipe);
        cp_async_fence();
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
    }

    float acc[Tm][Tn];
    CUTE_UNROLL
    for (int j = 0; j < Tm; ++j) {
        CUTE_UNROLL
        for (int v = 0; v < Tn; ++v) { acc[j][v] = 0.0f; }
    }

    int smem_pipe_read = 0;
    int smem_pipe_write = kStages - 1;

    CUTE_NO_UNROLL
    while (k_tile_count > -(kStages - 1)) {
        cp_async_wait<kStages - 2>();
        __syncthreads();

        load_a(k_tile_next, smem_pipe_write);
        load_b(k_tile_next, smem_pipe_write);
        cp_async_fence();

        // Unrolled, or every swizzled A address costs runtime index math. Left
        // alone ptxas then hoists all four groups' A loads and needs 167
        // registers for a 104-register loop; the launch bound holds it to the
        // four-block budget instead.
        const int rd = smem_pipe_read;
        CUTE_UNROLL
        for (int kg = 0; kg < bK; kg += 4) {
            float av[Tm][4];
            if constexpr (!TransA) {
                CUTE_UNROLL
                for (int j = 0; j < Tm; ++j) {
                    const float4 t =
                        *reinterpret_cast<const float4*>(&sA(arow(j), kg, rd));
                    av[j][0] = t.x; av[j][1] = t.y;
                    av[j][2] = t.z; av[j][3] = t.w;
                }
            } else {
                CUTE_UNROLL
                for (int kk = 0; kk < 4; ++kk) {
                    CUTE_UNROLL
                    for (int u = 0; u < Tm / 4; ++u) {
                        const float4 t =
                            *reinterpret_cast<const float4*>(&sA(arow(4 * u), kg + kk, rd));
                        av[4 * u][kk] = t.x; av[4 * u + 1][kk] = t.y;
                        av[4 * u + 2][kk] = t.z; av[4 * u + 3][kk] = t.w;
                    }
                }
            }

            if constexpr (!TransB) {
                CUTE_UNROLL
                for (int kk = 0; kk < 4; ++kk) {
                    float bv[Tn];
                    CUTE_UNROLL
                    for (int u = 0; u < Tn / 4; ++u) {
                        const float4 t =
                            *reinterpret_cast<const float4*>(&sB(brow(4 * u), kg + kk, rd));
                        bv[4 * u] = t.x; bv[4 * u + 1] = t.y;
                        bv[4 * u + 2] = t.z; bv[4 * u + 3] = t.w;
                    }
                    CUTE_UNROLL
                    for (int j = 0; j < Tm; ++j) {
                        CUTE_UNROLL
                        for (int v = 0; v < Tn; ++v) {
                            acc[j][v] = fmaf(av[j][kk], bv[v], acc[j][v]);
                        }
                    }
                }
            } else {
                // Holding all of B for the group as [Tn][4] is 32 floats at the
                // wide tile's Tn = 8, on top of av's 32 and the 64 accumulators,
                // and ptxas spills it. Streaming one n at a time costs the same
                // eight 16 B loads and keeps four B values live.
                CUTE_UNROLL
                for (int v = 0; v < Tn; ++v) {
                    const float4 t =
                        *reinterpret_cast<const float4*>(&sB(brow(v), kg, rd));
                    const float bv[4] = {t.x, t.y, t.z, t.w};
                    CUTE_UNROLL
                    for (int kk = 0; kk < 4; ++kk) {
                        CUTE_UNROLL
                        for (int j = 0; j < Tm; ++j) {
                            acc[j][v] = fmaf(av[j][kk], bv[kk], acc[j][v]);
                        }
                    }
                }
            }
        }

        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == kStages - 1) ? 0 : smem_pipe_read + 1;
    }

    const int ld_c = int(p.ldc);
    CUTE_UNROLL
    for (int j = 0; j < Tm; ++j) {
        const int m = m0 + arow(j);
        if (m >= M) { continue; }
        CUTE_UNROLL
        for (int v = 0; v < Tn; ++v) {
            const int n = n0 + brow(v);
            if (n >= N) { continue; }
            const int idx = m * ld_c + n;

            float x = p.alpha * acc[j][v];
            if (bias != nullptr) {
                x += bias[p.bias_row ? m : n];
            }
            if (p.beta != 0.0f) {
                x += p.beta * C[idx];
            }
            x = fminf(fmaxf(x, p.min_clip), p.max_clip);
            x = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, x);
            if (residual != nullptr) {
                x += residual[m * p.ldr + n];
            }
            C[idx] = x;
        }
    }
}

// ============================================================
// Launch
// ============================================================

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

template <bool TransA, bool TransB, bool Vectorized>
inline void launch_dense_h(const GemmParams& p, cudaStream_t stream)
{
    using namespace cute;
    constexpr int bM = 64, bN = 64, bK = 64, kStages = 3;
    constexpr auto la = gemm_smem_layout<!TransA>(Int<bM>{}, Int<bK>{}, Int<kStages>{});
    constexpr auto lb = gemm_smem_layout<TransB>(Int<bN>{}, Int<bK>{}, Int<kStages>{});
    constexpr int smem = int(cosize(la) + cosize(lb)) * int(sizeof(half_t));

    auto kernel = dense_gemm_h_kernel<TransA, TransB, Vectorized>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(p.m, bM), ceil_div(p.n, bN), gemm_batch_count(p));
    kernel<<<grid, kGemmThreads, smem, stream>>>(p);
}

template <bool TransA, bool TransB, bool Vectorized, bool Wide>
inline void launch_dense_f(const GemmParams& p, cudaStream_t stream)
{
    using namespace cute;
    constexpr int bM = 64, bN = Wide ? 128 : 64, bK = 16, kStages = Wide ? 2 : 3;
    constexpr int smem = (bM * bK + bN * bK) * kStages * int(sizeof(float));

    auto kernel = dense_gemm_f_kernel<TransA, TransB, Vectorized, Wide>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(p.m, bM), ceil_div(p.n, bN), gemm_batch_count(p));
    kernel<<<grid, kGemmThreads, smem, stream>>>(p);
}

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

/// Dense C = alpha*A*B + beta*C + bias, epilogue applied, optional residual.
/// Returns false when the request has no kernel (bad extents).
inline bool dense_gemm_launch(const GemmParams& p, int elem_size,
                              bool trans_a, bool trans_b, bool vectorized,
                              cudaStream_t stream)
{
    if (p.m <= 0 || p.n <= 0 || p.k <= 0) { return false; }
    // The 128-wide tile halves the A/B panel traffic, but its 30.7 KB of smem
    // costs a resident block, so it only pays once the grid can refill the
    // machine several times over.
    const int sm = detail::device_sm_count();
    const long long wide_blocks = static_cast<long long>((p.m + 63) / 64) *
                                  ((p.n + 127) / 128) * detail::gemm_batch_count(p);
    const bool wide = sm > 0 && wide_blocks >= 4LL * sm;
    detail::dispatch_dense(
        p, trans_a, trans_b, vectorized,
        [&]<bool TA, bool TB, bool V>() {
            if (elem_size == 2) {
                detail::launch_dense_h<TA, TB, V>(p, stream);
            } else if (wide) {
                detail::launch_dense_f<TA, TB, V, true>(p, stream);
            } else {
                detail::launch_dense_f<TA, TB, V, false>(p, stream);
            }
        });
    return true;
}

}  // namespace nnops::backend::cuda
