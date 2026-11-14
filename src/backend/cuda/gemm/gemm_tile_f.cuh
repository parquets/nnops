/// @file gemm_tile_f.cuh
/// @brief The f32 SIMT engine: a 64x128x16 UniversalFMA tile once the grid is
///        wide enough to refill the machine, 64x64x16 otherwise, streamed
///        through a cp.async pipeline.
///
/// The f32 tile swizzles its panel rows rather than padding them, which keeps a
/// four-block residency in 24 KB instead of 30 KB; a k-contiguous A panel can
/// take an explicit `padA` instead and leave the swizzle off.

#pragma once

#include "gemm_common.cuh"

#include <cuda_runtime.h>

namespace nnops::backend::cuda {

// ============================================================
// f32 SIMT kernel
// ============================================================

/// Everything the f32 kernel's shape depends on, as one searchable record.
///
/// Each field is a knob the benches sweep (`dense_gemm_f32_tile`), and each
/// distinct value is its own kernel, so a search multiplies instantiations: keep
/// the list short and only name the configurations that are being measured.
///  - `threads` and `tm` fix the thread tile: `bM/tm` threads along m, the rest
///    along n, and Tn falls out as `bN / (threads / (bM/tm))`. Tn is what costs
///    registers, so `threads` has to grow with the tile or the accumulators do.
///  - `blocks` is the resident-block target the launch bound has to satisfy; it
///    has to match what the shared-memory budget actually allows, and it is what
///    caps the register count (65536/(threads*blocks)).
///  - `padA` is the A row padding in floats, and only applies to a k-contiguous
///    A panel; 0 keeps the swizzle.
template <int ThreadsV, int TmV, int BMV, int BNV, int BKV, int StagesV,
          int BlocksV, int PadAV>
struct GemmTileF32
{
    static constexpr int threads = ThreadsV;
    static constexpr int tm = TmV;
    static constexpr int bM = BMV;
    static constexpr int bN = BNV;
    static constexpr int bK = BKV;
    static constexpr int stages = StagesV;
    static constexpr int blocks = BlocksV;
    static constexpr int padA = PadAV;
};

// 24576 B -> four blocks; 26624 B -> three. See the smem note in the kernel.
using kF32Narrow = GemmTileF32<128, 8, 64, 64, 16, 3, 4, 0>;
using kF32Wide = GemmTileF32<128, 8, 64, 128, 16, 2, 3, 4>;

template <class Tile, bool TransA, bool TransB, bool Vectorized>
__global__ __launch_bounds__(Tile::threads, Tile::blocks)
void dense_gemm_f_kernel(GemmParams p)
{
    using namespace cute;
    using Elem = float;

    constexpr int kThreads = Tile::threads;
    constexpr int bM = Tile::bM;
    constexpr int bN = Tile::bN;
    constexpr int bK = Tile::bK;
    constexpr int kStages = Tile::stages;
    static_assert(bK % 4 == 0, "every A load is one float4");
    static_assert(kStages >= 2, "the pipeline waits kStages - 2 deep");

    const int M = p.m;
    const int N = p.n;
    const int K = p.k;
    const int m0 = int(blockIdx.x) * bM;
    const int n0 = int(blockIdx.y) * bN;

    const int batch = detail::gemm_block_batch(p);
    const int split = detail::gemm_block_split(p);

    Elem const* A = static_cast<Elem const*>(p.a) +
                    detail::gemm_batch_offset(p, p.a_batch, batch);
    Elem const* B = static_cast<Elem const*>(p.b) +
                    detail::gemm_batch_offset(p, p.b_batch, batch);
    Elem const* bias = static_cast<Elem const*>(p.bias);
    Elem* C = static_cast<Elem*>(p.c) +
              detail::gemm_batch_offset(p, p.c_batch, batch);
    Elem const* residual = static_cast<Elem const*>(p.residual) +
                           detail::gemm_batch_offset(p, p.r_batch, batch);

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
    // k land on two banks. Both fixes cost something. Padding the row by one
    // 16 B vector keeps the row address loop invariant, so every load is an
    // immediate off a hoisted base, and buys that with 2 KB of shared memory.
    // The swizzle reproduces the pad's phase spread for free -- XORing the row
    // index into the 16 B vector index, bits 0..1 untouched so cp.async is still
    // 16 B -- but the address now depends on (row XOR k-block), and a runtime row
    // can only spell that as a full index recomputation per load. The wide tile
    // is down to three blocks a SM for register reasons anyway, so it pays for
    // the pad; the narrow one keeps its fourth block and swizzles. A panel wide
    // in the operand direction (>= 64 floats, two bank cycles per row) is already
    // conflict free and takes neither.
    constexpr int kPadA = (Tile::padA != 0 && !TransA) ? Tile::padA : 0;
    constexpr auto swz = Swizzle<3, 2, 3>{};
    constexpr auto sA_layout = [&] {
        if constexpr (TransA) {
            return make_layout(make_shape(Int<bM>{}, Int<bK>{}, Int<kStages>{}),
                               make_stride(Int<1>{}, Int<bM>{},
                                           Int<bM * bK>{}));
        } else if constexpr (kPadA != 0) {
            return make_layout(make_shape(Int<bM>{}, Int<bK>{}, Int<kStages>{}),
                               make_stride(Int<bK + kPadA>{}, Int<1>{},
                                           Int<(bK + kPadA) * bM>{}));
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
    constexpr int Tm = Tile::tm;
    constexpr int kThreadsM = bM / Tm;
    constexpr int kThreadsN = kThreads / kThreadsM;
    constexpr int Tn = bN / kThreadsN;
    static_assert(bM % Tm == 0 && Tm % 4 == 0, "A loads four rows at a time");
    static_assert(kThreads % kThreadsM == 0, "the block splits over m only");
    static_assert(bN % kThreadsN == 0 && Tn % 4 == 0, "B loads one float4 per n");

    const int tx = int(threadIdx.x) % kThreadsM;
    const int ty = int(threadIdx.x) / kThreadsM;
    auto arow = [&](int j) { return TransA ? tx * Tm + j : tx + kThreadsM * j; };
    auto brow = [&](int j) { return TransB ? ty + kThreadsN * j : ty * Tn + j; };

    // Every thread moves one 16 B vector per pass, so the thread layout is "the
    // block split over both axes": four threads along k when the panel is
    // k-contiguous, eight when the copy runs along m/n instead. A tiling then
    // covers `threads / n * 4` rows per pass, and a panel that is not a multiple
    // of that is overshot -- the copy addresses rows past the panel, which is the
    // next stage's buffer. A 64-wide B panel on 256 threads is exactly that
    // (32 threads x 4 = 128 rows for a 64-row panel), and it corrupts memory
    // rather than failing to launch.
    constexpr int kCopyRows = (kThreads / 8) * 4;
    static_assert(TransA || bM % (kThreads / 4) == 0,
                  "the A copy tiling overshoots the panel");
    static_assert(!TransA || bM % kCopyRows == 0,
                  "the A copy tiling overshoots the panel");
    static_assert(TransB || bN % kCopyRows == 0,
                  "the B copy tiling overshoots the panel");
    static_assert(!TransB || bN % (kThreads / 4) == 0,
                  "the B copy tiling overshoots the panel");

    // The swizzle is a composition of XORs on the linear offset, so it only tiles
    // a panel whose k-extent is a power of two. The pad takes any multiple of 4.
    static_assert(TransA || kPadA != 0 || (bK & (bK - 1)) == 0,
                  "the A swizzle needs a power-of-two bK");
    static_assert(!TransB || (bK & (bK - 1)) == 0,
                  "the B swizzle needs a power-of-two bK");
    auto copy_a = [&] {
        if constexpr (TransA) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<Int<kThreads / 8>, _8>>{},
                Layout<Shape<_4, _1>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<Int<kThreads / 4>, _4>, Stride<_4, _1>>{},
                Layout<Shape<_1, _4>>{});
        }
    }();
    auto copy_b = [&] {
        if constexpr (TransB) {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<Int<kThreads / 4>, _4>, Stride<_4, _1>>{},
                Layout<Shape<_1, _4>>{});
        } else {
            return make_tiled_copy(
                Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, Elem>{},
                Layout<Shape<Int<kThreads / 8>, _8>>{},
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

    int k_tile_count = 0;
    int k_tile_next = 0;
    detail::gemm_split_range(p, bK, split, k_tile_next, k_tile_count);

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

    // The stage index doubles as a shared-memory address offset, so leaving it a
    // running variable makes every operand load pay a runtime add. Unrolling the
    // stage cycle puts it into the layout as a constant and folds those adds into
    // immediates; ncu counted 195 ALU ops per k-tile per warp (15% of all issued
    // instructions) before either this or the A pad.
    auto k_body = [&]<int RD, int WR>() {
        cp_async_wait<kStages - 2>();
        __syncthreads();

        load_a(k_tile_next, WR);
        load_b(k_tile_next, WR);
        cp_async_fence();

        // Unrolled, or every swizzled A address costs runtime index math. Left
        // alone ptxas hoists all four groups' A loads and needs 167 registers
        // for a 104-register loop, which the launch bound then refuses.
        constexpr int rd = RD;
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
    };

    if constexpr (kStages == 2) {
        CUTE_NO_UNROLL
        while (k_tile_count > -1) {
            k_body.template operator()<0, 1>();
            if (k_tile_count > -1) { k_body.template operator()<1, 0>(); }
        }
    } else {
        CUTE_NO_UNROLL
        while (k_tile_count > -2) {
            k_body.template operator()<0, 2>();
            if (k_tile_count > -2) { k_body.template operator()<1, 0>(); }
            if (k_tile_count > -2) { k_body.template operator()<2, 1>(); }
        }
    }

    // Straight from the accumulators, no staging tile: a thread's Tm rows x Tn
    // columns are contiguous along n, so an instruction already fills a whole
    // 128 B segment per row across the block -- and staging the CTA tile over the
    // dead cp.async panels to coalesce it further was measured at -13% (64x128,
    // two passes) and -0.4% (64x64, one pass): the epilogue is exposed at the end
    // of a block's life, so the round trip and the barrier cost more than the
    // half-filled sectors they remove.
    // A split's block stops at its accumulator: the sum, the bias and the
    // activation all belong to the reduce kernel, which sees every plane.
    float* part = nullptr;
    if (p.k_splits > 1) {
        const int64_t plane = static_cast<int64_t>(M) * N;
        part = p.partial + (static_cast<int64_t>(batch) * p.k_splits + split) * plane;
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
            if (part != nullptr) {
                part[static_cast<int64_t>(m) * N + n] = acc[j][v];
                continue;
            }

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

namespace detail {

template <class Tile, bool TransA, bool TransB, bool Vectorized>
inline void launch_dense_f(const GemmParams& p, cudaStream_t stream)
{
    using namespace cute;
    constexpr int bM = Tile::bM, bN = Tile::bN, bK = Tile::bK, kStages = Tile::stages;
    constexpr int kPadA = (Tile::padA != 0 && !TransA) ? Tile::padA : 0;
    constexpr int smem =
        (bM * (bK + kPadA) + bN * bK) * kStages * int(sizeof(float));
    static_assert(smem * Tile::blocks <= kSmemPerSm, "the block target does not fit");

    auto kernel = dense_gemm_f_kernel<Tile, TransA, TransB, Vectorized>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    dim3 grid(ceil_div(p.m, bM), ceil_div(p.n, bN),
              gemm_batch_count(p) * p.k_splits);
    kernel<<<grid, Tile::threads, smem, stream>>>(p);
}

}  // namespace detail

}  // namespace nnops::backend::cuda
