/// @file gemm_tile_h.cuh
/// @brief The f16/bf16 tensor-core engine: an SM80 16x8x16 MMA tile with f32
///        accumulators, streamed through a cp.async pipeline.
///
/// The gmem->smem copy is a 16 B vector load per thread whenever every row pitch
/// is 16 B aligned and the tile is fully in bounds; the alternative kernel
/// variant fills shared memory scalar-wise with a bounds check, which is also
/// what the edge tiles of an aligned call fall back to. That split is why the
/// smem layout follows the gmem order of each operand: a 16 B cp.async must land
/// contiguous in shared memory, and CuTe's LDSM atoms are the ones paired with
/// each of the two layouts (k-major -> LDSM_N, operand-major -> LDSM_T).
///
/// A tile is `Tile::threads` threads over a `bM x bN` accumulator block, and
/// `Groups` of them run side by side in one CTA, which is how the paired launch
/// halves the resident-block count in exchange for a second warp per scheduler.

#pragma once

#include "gemm_common.cuh"

#include <cuda_runtime.h>
#include <type_traits>

namespace nnops::backend::cuda {

// ============================================================
// f16 tensor-core kernel
// ============================================================

/// The f16 tile, as one searchable record, for the same reason as the f32 one:
/// every distinct tile is its own kernel.
///  - `wm` x `wn` is the warp grid and `mmaM` x `mmaN` the tile one warp's atom
///    covers; together they fix the MMA, and `threads` follows from the first.
///  - `bM` x `bN` is the CTA tile, which the MMA tile may divide into several
///    pieces. Those pieces replicate the accumulator fragment -- `partition_C`
///    hands a thread one per piece -- so this is what buys a wider tile without
///    a wider warp.
///  - `group` sizes the L2 raster, in tiles along m.
template <int ThreadsV, int BMV, int BNV, int BKV, int StagesV, int WMV, int WNV,
          int MmaMV, int MmaNV, int GroupV>
struct GemmTileH
{
    static constexpr int threads = ThreadsV;
    static constexpr int bM = BMV;
    static constexpr int bN = BNV;
    static constexpr int bK = BKV;  // 64-multiples take the swizzle, the rest a pad
    static constexpr int stages = StagesV;
    static constexpr int wm = WMV;
    static constexpr int wn = WNV;
    static constexpr int mmaM = MmaMV;
    static constexpr int mmaN = MmaNV;
    static constexpr int group = GroupV;
    static_assert(wm * wn * 32 == threads, "one warp per warp tile");
    static_assert(mmaM % 16 == 0 && mmaN % 8 == 0, "the warp tile is an atom multiple");
    static_assert(bM % (wm * mmaM) == 0 && bN % (wn * mmaN) == 0,
                  "the MMA tile divides the CTA tile");
    static_assert(bK % 8 == 0, "a 16 B vector does not divide the k tile");
};

/// 64x64x32 on four warps: three stages fit three blocks a SM, four only two,
/// and the third block is worth more than the extra stage everywhere. 27 KB, and
/// 39 KB paired. The k tile is half the swizzle atom wide, which costs a padded
/// smem layout (see `gemm_smem_layout`) and buys a pipeline that keeps two groups
/// in flight instead of draining the whole 16 KB panel every k tile: 2-11% over
/// the 64-wide two-stage tile on every shape measured, against a 6% loss at four
/// stages. The L2 raster walks four m tiles rather than eight: the narrower band
/// is 4% of the 4 MB L2 at 512^3 against 7% at eight.
using kF16Narrow = GemmTileH<128, 64, 64, 32, 3, 2, 2, 32, 32, 4>;
/// No longer dispatched: the paired CTA (`dense_gemm_f16_groups`) measured ahead
/// of it on every shape it used to win. Kept as the bench's comparison arm.
using kF16Wide = GemmTileH<128, 64, 128, 64, 2, 2, 2, 32, 32, 8>;

namespace detail {

/// The gmem->smem copy for one panel: a 16 B vector per thread per pass, so the
/// thread layout splits the panel eight halves deep along whichever axis is
/// contiguous, and the passes run over the other axis. A tiling whose thread
/// count does not divide the panel walks off its end -- into the next stage's
/// buffer rather than into a fault -- so the shape is checked here instead.
template <class Elem, int Threads, int Rows, int Cols, bool ContigCols>
constexpr auto gemm_copy_half()
{
    if constexpr (ContigCols) {
        constexpr int t_c = Cols / 8;
        constexpr int t_o = Threads / t_c;
        static_assert(Cols % 8 == 0, "a 16 B vector does not divide the panel");
        static_assert(Threads % t_c == 0 && Rows % t_o == 0,
                      "the copy tiling overshoots the panel");
        return cute::make_tiled_copy(
            cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                            Elem>{},
            cute::Layout<cute::Shape<cute::Int<t_o>, cute::Int<t_c>>>{},
            cute::Layout<cute::Shape<cute::_1, cute::_8>>{});
    } else {
        constexpr int t_r = Rows / 8;
        constexpr int t_c = Threads / t_r;
        static_assert(Rows % 8 == 0, "a 16 B vector does not divide the panel");
        static_assert(Threads % t_r == 0 && Cols % t_c == 0,
                      "the copy tiling overshoots the panel");
        return cute::make_tiled_copy(
            cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>,
                            Elem>{},
            cute::Layout<cute::Shape<cute::Int<t_r>, cute::Int<t_c>>>{},
            cute::Layout<cute::Shape<cute::_8, cute::_1>>{});
    }
}

/// The 16x8x16 tensor-core atom for a 2-byte float. f16 and bf16 have the same
/// MMA shape; only the operand format differs.
template <class Elem>
using gemm_mma_atom = std::conditional_t<std::is_same_v<Elem, cute::bfloat16_t>,
                                         cute::SM80_16x8x16_F32BF16BF16F32_TN,
                                         cute::SM80_16x8x16_F32F16F16F32_TN>;

}  // namespace detail

template <class Tile, class Elem, bool TransA, bool TransB, bool Vectorized,
          int Groups = 1>
__global__ __launch_bounds__(Tile::threads * Groups)
void dense_gemm_h_kernel(GemmParams p)
{
    using namespace cute;
    using In = Elem;
    using Out = Elem;

    constexpr int bM = Tile::bM;
    constexpr int bN = Tile::bN;
    constexpr int bK = Tile::bK;
    constexpr int kStages = Tile::stages;
    constexpr int kThreads = Tile::threads;
    constexpr int kCtaThreads = kThreads * Groups;

    // A CTA is `Groups` copies of the four-warp tile side by side along n. The
    // 2x2 warp grid over 32x32 warp tiles is the one arrangement the ldmatrix
    // copy helpers accept, so the way to more warps in a block is more tiles in
    // it: each group keeps its own accumulators, its own C tile and its own B
    // panel, and shares the A panel that all of them read.
    const int wg = int(threadIdx.x) / kThreads;
    const int lane = int(threadIdx.x) % kThreads;

    const int M = p.m;
    const int N = p.n;
    const int K = p.k;

    // The operand traffic, not the tensor pipe, is what limits this kernel: ncu
    // at 4096^3 puts HMMA at 46% and DRAM at 2.3 GB read, which is the whole
    // runtime. Dispatched in blockIdx order a block column re-reads all of A once
    // per n tile -- sixty-four times here, L2 being 4 MB against a 33 MB operand.
    // Walking a square group of tiles instead keeps only the group's k-slice, a
    // few hundred KB, resident long enough to be shared by ten blocks.
    const int num_m = (M + bM - 1) / bM;
    const int num_n = (N + bN - 1) / bN;
    const int num_nc = (num_n + Groups - 1) / Groups;
    constexpr int kGroup = Tile::group;
    const int group_span = kGroup * num_nc;
    const int gid = int(blockIdx.x) / group_span;
    const int within = int(blockIdx.x) - gid * group_span;
    const int m_span = (num_m - gid * kGroup) < kGroup ? (num_m - gid * kGroup) : kGroup;
    const int tile_m = gid * kGroup + within % m_span;
    // The last CTA of an odd n axis has a group with no tile; its bounds checks
    // carry it through the copy and the epilogue without a store.
    const int tile_n = (within / m_span) * Groups + wg;

    const int m0 = tile_m * bM;
    const int n0 = tile_n * bN;

    const int batch = detail::gemm_block_batch(p);
    const int split = detail::gemm_block_split(p);

    In const* A = static_cast<In const*>(p.a) +
                  detail::gemm_batch_offset(p, p.a_batch, batch);
    In const* B = static_cast<In const*>(p.b) +
                  detail::gemm_batch_offset(p, p.b_batch, batch);
    In const* bias = static_cast<In const*>(p.bias);
    Out* C = static_cast<Out*>(p.c) +
             detail::gemm_batch_offset(p, p.c_batch, batch);
    Out const* residual = static_cast<Out const*>(p.residual) +
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
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N),
                            make_stride(int(p.ldc), Int<1>{}));

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(tile_m, tile_n, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    constexpr auto sA_layout = detail::gemm_smem_layout<!TransA>(
        Int<bM>{}, Int<bK>{}, Int<kStages>{});
    constexpr auto sB_layout = detail::gemm_smem_layout<TransB>(
        Int<bN>{}, Int<bK>{}, Int<kStages>{});

    constexpr int kSmemA = int(cosize(sA_layout));
    constexpr int kSmemB = int(cosize(sB_layout));  // one group's panel

    extern __shared__ char gemm_smem[];
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<In*>(gemm_smem)), sA_layout);
    Tensor sB = make_tensor(
        make_smem_ptr(reinterpret_cast<In*>(gemm_smem + (kSmemA + wg * kSmemB) * 2)),
        sB_layout);

    TiledMMA mma = make_tiled_mma(detail::gemm_mma_atom<Elem>{},
                                  Layout<Shape<Int<Tile::wm>, Int<Tile::wn>, _1>>{},
                                  cute::Tile<Int<Tile::mmaM>, Int<Tile::mmaN>, _16>{});

    // The A panel is shared, so the whole CTA copies it; a B panel belongs to one
    // group, so that group's threads copy it with their own slice indices.
    auto copy_a = detail::gemm_copy_half<Elem, kCtaThreads, bM, bK, !TransA>();
    auto copy_b = detail::gemm_copy_half<Elem, kThreads, bN, bK, TransB>();

    ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA);
    Tensor tAsA = thr_copy_a.partition_D(sA);

    ThrCopy thr_copy_b = copy_b.get_slice(lane);
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
        for (int i = lane; i < bN * bK; i += kThreads) {
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

    ThrMMA thr_mma = mma.get_slice(lane);
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
    ThrCopy s2r_thr_copy_a = s2r_copy_a.get_slice(lane);
    Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
    Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

    TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
    ThrCopy s2r_thr_copy_b = s2r_copy_b.get_slice(lane);
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

    // A split's block stops at its accumulator: the sum, the bias and the
    // activation all belong to the reduce kernel, which sees every plane.
    float* part = nullptr;
    if (p.k_splits > 1) {
        const int64_t plane = static_cast<int64_t>(M) * N;
        part = p.partial + (static_cast<int64_t>(batch) * p.k_splits + split) * plane;
    }

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        const int m = int(get<0>(coord));
        const int n = int(get<1>(coord));
        if (m >= M || n >= N) { continue; }
        if (part != nullptr) {
            part[static_cast<int64_t>(m) * N + n] = float(tCrC(i));
            continue;
        }

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

namespace detail {

template <class Tile, class Elem, bool TransA, bool TransB, bool Vectorized,
          int Groups = 1>
inline void launch_dense_h(const GemmParams& p, cudaStream_t stream)
{
    using namespace cute;
    constexpr int bM = Tile::bM, bN = Tile::bN, bK = Tile::bK;
    constexpr auto la = gemm_smem_layout<!TransA>(Int<bM>{}, Int<bK>{}, Int<Tile::stages>{});
    constexpr auto lb = gemm_smem_layout<TransB>(Int<bN>{}, Int<bK>{}, Int<Tile::stages>{});
    constexpr int smem = int(cosize(la) + Groups * cosize(lb)) * int(sizeof(Elem));
    static_assert(smem <= kSmemPerSm, "one block has to fit");

    auto kernel = dense_gemm_h_kernel<Tile, Elem, TransA, TransB, Vectorized, Groups>;
    static const cudaError_t cfg = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    // One flat tile index: the kernel turns it into an L2-grouped (m, n) pair.
    dim3 grid(ceil_div(p.m, bM) * ceil_div(p.n, bN * Groups), 1,
              gemm_batch_count(p) * p.k_splits);
    kernel<<<grid, Tile::threads * Groups, smem, stream>>>(p);
}

}  // namespace detail

}  // namespace nnops::backend::cuda
