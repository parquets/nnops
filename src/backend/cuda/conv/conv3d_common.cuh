#pragma once
/// @file conv3d_common.cuh
/// @brief The 3-D convolution family's kernels: one implicit GEMM over the
///        output-voxel axis, with the transposed pose as a gather predicate.
///
/// Conv3D and TransposeConv3D are the same product. Both write an output voxel
/// from a reduction over (ic, kd, kh, kw); the difference is which input voxel
/// a (voxel, tap) pair names — Conv3D strides forward from the output position,
/// TransposeConv3D divides (o + pad − k·dil) by the stride and drops the taps
/// that do not divide. Everything else — the operand walks, the panel fills,
/// the epilogue — is shared, so it is one kernel with a template flag.
///
/// M is the output-voxel axis, N the output channels of one group, K the
/// reduction IC/G·KD·KH·KW. One block owns 64 voxels by 64 channels and steps
/// the reduction a stage of k at a time through a cp.async ring, so the gather
/// of stage t+1 runs under the MMA of stage t.
///
/// The mainloops are the GEMM engine's, not this file's: `f32_gemm_tile` on a
/// padded panel pair for f32, `f16_gemm_tile` on the swizzled pair for the
/// 16-bit dtypes. Both take the panels in their own layouts, which is what lets
/// a gathered fill sit where the dense GEMM's cp.async copy sits.
///
/// **The gather is the whole cost** — with it stubbed out this kernel runs at
/// 6253 GFLOPS against 3144 with it — and three things keep it off the critical
/// path. A thread owns one k column, one tap's channel run, so the tap decodes
/// once a stage rather than once an element; a division by a runtime value is a
/// twenty-instruction sequence on this ISA and the element-at-a-time form spent
/// two of them on every element of the panel. The run is one 16 B vector — four
/// channels for f32, eight for the 16-bit dtypes — which is exactly what a tap's
/// channels are in NDHWC, so one address computation and one bounds test cover
/// the vector. And a vector tap is either wholly inside the volume or wholly
/// outside it, so the test is per (voxel, tap) at the vector's granularity, and
/// the copy is one cp.async or one zero.

#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/matmul/matmul_epilogue.cuh" // gemm_apply_activation
#include "backend/cuda/matmul/matmul_f16.cuh"
#include "backend/cuda/matmul/matmul_f32.cuh"
#include "backend/cuda/matmul/ptx.h"
#include "nnops/core/status.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda::conv3d {

/// The f32 tile: 128 voxels by 64 channels, 4x8 a lane, eight warps, 16 k a
/// stage, two stages. Conv2D's lane tile, one warp row wider: a 3-D output has
/// the voxel axis to itself, so the extra warps go there and the block reaches
/// 24 resident warps at 29 KB of panels.
using C3ShapeF = F32TileShape<4, 8, 4, 2, 16, 2, 4, 8, 4, 3>;
/// The transposed pose reads the same tile one warp row taller: its gather
/// divides the source index, which costs the same per voxel however tall the
/// tile, so the extra rows amortize it (measured: 2704 -> 3497 GFLOPS on a 2x
/// upsampling, against 3102 -> 2677 the other way for the forward pose).
using C3ShapeFT = F32TileShape<4, 8, 4, 2, 16, 2, 4, 8, 4, 3>;
/// The 16-bit tile: the same 64 by 64 output block, eight warps, 64 k a stage on
/// the m16n8k16 atom.
using C3ShapeH = F16TileShape<32, 16, 2, 4, 64, 2, 1, false, 3>;

/// The tile an element type runs: the host's grid and the kernel's block both
/// read it here, so a shape is one place.
template <class Elem, bool TRANS> struct C3ShapeOf {
    using type = std::conditional_t<sizeof(Elem) == 4,
                                    std::conditional_t<TRANS, C3ShapeFT, C3ShapeF>, C3ShapeH>;
};
/// The tile a (dtype, pose) pair runs.
template <class Elem, bool TRANS> using C3Shape = typename C3ShapeOf<Elem, TRANS>::type;
/// The tile's geometry, for the host's grid.
template <class Elem, bool TRANS> constexpr int c3_bm()
{
    return C3Shape<Elem, TRANS>::BM;
}
template <class Elem, bool TRANS> constexpr int c3_bn()
{
    return C3Shape<Elem, TRANS>::BN;
}
/// The largest row block either pose uses: the row table is sized to it.
constexpr int kC3Rows = 128;
static_assert(C3ShapeF::BM <= kC3Rows && C3ShapeFT::BM <= kC3Rows &&
                  C3ShapeH::BM <= kC3Rows, "the row table must hold a tile");

/// One 3-D convolution's geometry, strides and epilogue. The input points at its
/// batch slice and the output at its batch·group slice; the filter at the
/// group's base. Every gather offset is 32-bit, which the host guarantees by
/// refusing a slice whose span leaves the int range.
struct C3Params {
    int id = 0, ih = 0, iw = 0, ic = 0;
    int od = 0, oh = 0, ow = 0, oc = 0;
    int kd = 1, kh = 1, kw = 1;
    int sd = 1, sh = 1, sw = 1;
    int dd = 1, dh = 1, dw = 1;
    int pd = 0, ph = 0, pw = 0;
    int icg = 1, ocg = 1, groups = 1;

    int64_t m = 0; ///< OD·OH·OW: the M axis of one batch·group slice
    int64_t in_batch = 0, in_d = 0, in_h = 0, in_w = 0, in_c = 0;
    int64_t out_batch = 0, out_d = 0, out_h = 0, out_w = 0, out_c = 0;
    int64_t w_kd = 0, w_kh = 0, w_kw = 0, w_ic = 0, w_oc = 0;
    int64_t w_group = 0; ///< element stride between groups

    const void* in = nullptr;
    const void* w = nullptr;
    void* out = nullptr;
    const void* bias = nullptr;

    float min_clip = -3.4e38f, max_clip = 3.4e38f;
    float ep_alpha = 0.0f, ep_beta = 1.0f;
    EpilogueActivateType ep = EpilogueActivateType::None;
    int add_to = 0, pre_act = 0;
    /// 16 B copies are legal: the activation's tap run and the filter's output
    /// channel run are each contiguous and aligned.
    int fill_vec = 0, w_vec = 0;
};

/// The tile's voxel coordinates, one a row of the tile, filled once a block:
/// the gather walks rows the compute does not own, so they live in shared
/// memory rather than in the compute's registers.
///
/// `d0/h0/w0` carry the pose's own head of the source index — Conv3D's
/// o·s − p, TransposeConv3D's o + p — so a tap only adds or subtracts its own
/// k·d, and `inbase` folds the forward pose's base in once a row. The tap part
/// is the calling thread's (one tap a thread), so the two are added, never
/// counted twice. `obase` is the output element's offset. `ok` is false past the
/// end of the slice, which is the block's last tile of voxels.
struct C3RowTable {
    int d0[kC3Rows], h0[kC3Rows], w0[kC3Rows];
    int inbase[kC3Rows], obase[kC3Rows], ok[kC3Rows];
};

template <bool TRANS>
__device__ __forceinline__ void c3_row_table(C3RowTable& t, const C3Params& p, int64_t m0,
                                             int tid) noexcept
{
    if (tid >= kC3Rows) {
        return;
    }
    const int64_t v = m0 + tid;
    if (v >= p.m) {
        t.d0[tid] = 0;
        t.h0[tid] = 0;
        t.w0[tid] = 0;
        t.inbase[tid] = 0;
        t.obase[tid] = 0;
        t.ok[tid] = 0;
        return;
    }
    const int64_t ow = v % p.ow;
    const int64_t r = v / p.ow;
    const int od = static_cast<int>(r / p.oh);
    const int oh = static_cast<int>(r % p.oh);
    t.d0[tid] = static_cast<int>(TRANS ? od + p.pd : od * p.sd - p.pd);
    t.h0[tid] = static_cast<int>(TRANS ? oh + p.ph : oh * p.sh - p.ph);
    t.w0[tid] = static_cast<int>(TRANS ? int(ow) + p.pw : int(ow) * p.sw - p.pw);
    t.inbase[tid] = TRANS ? 0
                          : static_cast<int>(int64_t(t.d0[tid]) * p.in_d +
                                             int64_t(t.h0[tid]) * p.in_h +
                                             int64_t(t.w0[tid]) * p.in_w);
    t.obase[tid] = static_cast<int>(int64_t(od) * p.out_d + int64_t(oh) * p.out_h +
                                    int64_t(ow) * p.out_w);
    t.ok[tid] = 1;
}

/// The k index (tap·IC/G + ic) of the reduction, decoded once per tap.
/// Conv3D's DHWIO filter is in this order; TransposeConv3D's is not, and walks
/// its own strides from the same three tap indices.
__device__ __forceinline__ void c3_decode(const C3Params& p, int k, int& kd, int& kh, int& kw,
                                          int& ic) noexcept
{
    const int tap = k / p.icg;
    ic = k - tap * p.icg;
    kd = tap / (p.kh * p.kw);
    const int r = tap - kd * p.kh * p.kw;
    kh = r / p.kw;
    kw = r - kh * p.kw;
}

/// The three panel indexers the fills write through, in the layouts the two GEMM
/// tiles read. f32's panels are padded `[m][k]` / `[k][n]`; the 16-bit ones are
/// swizzled, which costs no padding bytes at all.
template <int LDA> struct C3PanelF {
    float* base;
    __device__ __forceinline__ float& operator()(int row, int col) const noexcept
    {
        return base[row * LDA + col];
    }
};

template <class S, class Elem> struct C3PanelA {
    Elem* base;
    __device__ __forceinline__ Elem& operator()(int row, int col) const noexcept
    {
        return base[f16_swz<S>(row, col)];
    }
};

template <class S, class Elem> struct C3PanelB {
    Elem* base;
    __device__ __forceinline__ Elem& operator()(int row, int col) const noexcept
    {
        return base[f16_swz_n<S>(row, col)];
    }
};

/// Gather one A stage: BM voxels by BK k, one thread per (tap, channel run)
/// walking the voxel rows, so the tap decodes once and its bounds test is the
/// only per-row work. A 16 B run of `CH` channels starting at `col` is `CH`
/// contiguous panel elements, which is what makes the vector copy legal; a
/// geometry whose group holds fewer than `CH` channels — or a run that would
/// straddle two taps — takes the element path, where a thread owns a single k
/// and the tap still decodes once.
template <class Elem, int BK, int BM, int THREADS, bool TRANS, class At>
__device__ __forceinline__ void c3_fill_a(const C3Params& p, At at, const C3RowTable& rows,
                                          int64_t k0, const Elem* in, int tid, int64_t K) noexcept
{
    constexpr int CH = 16 / int(sizeof(Elem)); // one 16 B vector of k
    using Vec = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;
    const int in_d = static_cast<int>(p.in_d), in_h = static_cast<int>(p.in_h);
    const int in_w = static_cast<int>(p.in_w);
    const bool vec = p.fill_vec != 0;

    // One k a thread on the element path, one channel run a thread on the vector
    // path; `span` is how many panel elements that thread writes a row.
    const int col = vec ? tid % (BK / CH) : tid % BK;
    const int first = vec ? col * CH : col;
    const int span = vec ? CH : 1;
    const int rowsPerPass = THREADS / (vec ? (BK / CH) : BK);

    const int k = static_cast<int>(k0) + first;
    int kd = 0, kh = 0, kw = 0, ic = 0;
    c3_decode(p, k, kd, kh, kw, ic);
    const bool k_ok = k < K;
    const int dd = kd * p.dd, dh = kh * p.dh, dw = kw * p.dw;
    // The run's first channel, and the tap's share of the *forward* source
    // offset. The transposed pose folds the tap into the source index itself,
    // so it takes the channel part alone.
    const int ch_off = ic;
    const int tap_off = dd * in_d + dh * in_h + dw * in_w;
    const Vec zero = {};

    for (int row = tid / (vec ? (BK / CH) : BK); row < BM; row += rowsPerPass) {
        int off = -1;
        if (k_ok && rows.ok[row]) {
            if constexpr (!TRANS) {
                const int id = rows.d0[row] + dd;
                const int ih = rows.h0[row] + dh;
                const int iw = rows.w0[row] + dw;
                if (unsigned(id) < unsigned(p.id) && unsigned(ih) < unsigned(p.ih) &&
                    unsigned(iw) < unsigned(p.iw)) {
                    off = rows.inbase[row] + tap_off + ch_off;
                }
            } else {
                // o + p − k·d must divide by the stride; the quotient is i.
                const int nd = rows.d0[row] - dd;
                const int nh = rows.h0[row] - dh;
                const int nw = rows.w0[row] - dw;
                if (nd % p.sd == 0 && nh % p.sh == 0 && nw % p.sw == 0) {
                    const int id = nd / p.sd, ih = nh / p.sh, iw = nw / p.sw;
                    if (unsigned(id) < unsigned(p.id) && unsigned(ih) < unsigned(p.ih) &&
                        unsigned(iw) < unsigned(p.iw)) {
                        off = id * in_d + ih * in_h + iw * in_w + ch_off;
                    }
                }
            }
        }
        if (vec && off >= 0) {
            cp_async_ca_16B(&at(row, first), in + off);
            continue;
        }
        for (int c = 0; c < span; ++c) {
            at(row, first + c) = off >= 0 ? in[off + c] : static_cast<Elem>(0);
        }
    }
}

/// Gather one B stage: BK k by BN output channels. A thread owns one k, so its
/// tap decodes once and the channel runs walk. Conv3D's DHWIO filter has the
/// output channel innermost, so a run is one 16 B copy; TransposeConv3D's order
/// puts the tap axes innermost and its run is a stride, which the host reports
/// as `w_vec = 0` and the fill walks element by element.
template <class Elem, int BK, int BN, int THREADS, class At>
__device__ __forceinline__ void c3_fill_b(const C3Params& p, At at, int n0, int64_t k0,
                                          const Elem* w, int tid, int64_t K) noexcept
{
    constexpr int CH = 16 / int(sizeof(Elem));
    constexpr int kCols = BN / CH;
    constexpr int kRowsPerPass = THREADS / kCols;
    using Vec = std::conditional_t<sizeof(Elem) == 2, uint4, float4>;
    const Vec zero = {};

    // The channel run is the fast axis, so a warp's copies are one contiguous
    // run of the filter — the row is what walks, one k at a time.
    const int nr = tid % kCols;
    const int n = n0 + nr * CH;
    for (int krow = tid / kCols; krow < BK; krow += kRowsPerPass) {
        const int k = static_cast<int>(k0) + krow;
        int kd = 0, kh = 0, kw = 0, ic = 0;
        c3_decode(p, k, kd, kh, kw, ic);
        if (k < K && p.w_vec && n + CH <= p.ocg) {
            cp_async_ca_16B(&at(krow, nr * CH),
                            w + int64_t(ic) * p.w_ic + int64_t(n) * p.w_oc +
                                int64_t(kd) * p.w_kd + int64_t(kh) * p.w_kh +
                                int64_t(kw) * p.w_kw);
            continue;
        }
        const int koff = k < K ? static_cast<int>(ic * p.w_ic + kd * p.w_kd + kh * p.w_kh +
                                                 kw * p.w_kw)
                               : 0;
        for (int c = 0; c < CH; ++c) {
            at(krow, nr * CH + c) = (k < K && n + c < p.ocg)
                                        ? w[koff + int64_t(n + c) * p.w_oc]
                                        : static_cast<Elem>(0);
        }
    }
}

/// The f32 kernel: the SIMT tile over a padded panel ring.
template <class S, bool TRANS>
__global__ __launch_bounds__(S::threads, S::MinBlocks)
void conv3d_kernel_f(C3Params p)
{
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;

    extern __shared__ __align__(16) unsigned char c3_smem[];
    float* const sAbase = reinterpret_cast<float*>(c3_smem);
    float* const sBbase = sAbase + ST * S::a_panel(false);
    __shared__ C3RowTable rows;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int warp_m = warp % S::wm;
    const int warp_n = warp / S::wm;

    const int m0 = static_cast<int>(blockIdx.x) * BM;
    const int n0 = static_cast<int>(blockIdx.y) * BN;
    const int64_t g = int64_t(blockIdx.z) % p.groups;
    const int64_t b = int64_t(blockIdx.z) / p.groups;

    const float* in = static_cast<const float*>(p.in) + b * p.in_batch + g * p.icg * p.in_c;
    const float* W = static_cast<const float*>(p.w) + g * p.w_group;
    float* out = static_cast<float*>(p.out) + b * p.out_batch + g * p.ocg * p.out_c;
    const float* bias =
        p.bias != nullptr ? static_cast<const float*>(p.bias) + g * p.ocg : nullptr;

    c3_row_table<TRANS>(rows, p, m0, tid);
    __syncthreads();

    const int64_t K = int64_t(p.icg) * p.kd * p.kh * p.kw;
    const int kt_end = static_cast<int>((K + BK - 1) / BK);

    auto fill = [&](int kt, int stage) {
        c3_fill_a<float, BK, BM, S::threads, TRANS>(p, C3PanelF<S::LDA>{sAbase + stage * S::a_panel(false)},
                                                    rows, int64_t(kt) * BK, in, tid, K);
        c3_fill_b<float, BK, BN, S::threads>(p, C3PanelF<S::LDN>{sBbase + stage * S::b_panel(false)},
                                             n0, int64_t(kt) * BK, W, tid, K);
        cp_async_commit_group();
    };

    float acc[S::LaneRow][S::LaneCol];
#pragma unroll
    for (int r = 0; r < S::LaneRow; ++r) {
#pragma unroll
        for (int c = 0; c < S::LaneCol; ++c) {
            acc[r][c] = 0.0f;
        }
    }

    // Prime every stage but one; the loop keeps ST-1 groups in flight, so
    // `wait_group<ST-2>` retires exactly the tile about to be read.
#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(t, t % ST);
    }
    for (int kt = 0; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);
        f32_gemm_tile<S, false, false>(sAbase + stage * S::a_panel(false),
                                       sBbase + stage * S::b_panel(false), warp_m, warp_n, lane,
                                       acc);
    }

    // A lane is four lane-rows by eight lane-cols: its accumulator row r is the
    // row eight apart, its column c the next column of the lane's row.
    const int lrow = lane % 8;
    const int lcol = lane / 8;
    const int row0 = warp_m * S::warpRows + lrow;
    const int col0 = warp_n * S::warpCols + lcol * S::LaneCol;
#pragma unroll
    for (int r = 0; r < S::LaneRow; ++r) {
        const int row = row0 + r * 8;
        const int m = m0 + row;
        if (m >= p.m) {
            continue;
        }
        const int64_t base = rows.obase[row];
#pragma unroll
        for (int c = 0; c < S::LaneCol; ++c) {
            const int n = n0 + col0 + c;
            if (n >= p.ocg) {
                continue;
            }
            float v = acc[r][c];
            float* dst = out + base + int64_t(n) * p.out_c;
            if (bias != nullptr) {
                v += bias[n];
            }
            if (p.add_to && p.pre_act) {
                v += *dst;
            }
            v = fminf(fmaxf(v, p.min_clip), p.max_clip);
            v = gemm_apply_activation(p.ep, p.ep_alpha, p.ep_beta, v);
            if (p.add_to && !p.pre_act) {
                v += *dst;
            }
            *dst = v;
        }
    }
}

/// The 16-bit kernel: the m16n8k16 tile over a swizzled panel ring.
template <class S, class Elem, bool TRANS>
__global__ __launch_bounds__(S::threads, S::MinBlocks)
void conv3d_kernel_h(C3Params p)
{
    constexpr int BM = S::BM, BN = S::BN, BK = S::BK, ST = S::stages;

    extern __shared__ __align__(16) unsigned char c3_smem[];
    Elem* const sAbase = reinterpret_cast<Elem*>(c3_smem);
    Elem* const sBbase = sAbase + ST * S::kAMax;
    __shared__ C3RowTable rows;

    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;
    const int warp_m = warp % S::wm;
    const int warp_n = warp / S::wm;

    const int m0 = static_cast<int>(blockIdx.x) * BM;
    const int n0 = static_cast<int>(blockIdx.y) * BN;
    const int64_t g = int64_t(blockIdx.z) % p.groups;
    const int64_t b = int64_t(blockIdx.z) / p.groups;

    const Elem* in = static_cast<const Elem*>(p.in) + b * p.in_batch + g * p.icg * p.in_c;
    const Elem* W = static_cast<const Elem*>(p.w) + g * p.w_group;
    Elem* out = static_cast<Elem*>(p.out) + b * p.out_batch + g * p.ocg * p.out_c;
    const Elem* bias = p.bias != nullptr ? static_cast<const Elem*>(p.bias) + g * p.ocg : nullptr;

    c3_row_table<TRANS>(rows, p, m0, tid);
    __syncthreads();

    const int64_t K = int64_t(p.icg) * p.kd * p.kh * p.kw;
    const int kt_end = static_cast<int>((K + BK - 1) / BK);

    auto fill = [&](int kt, int stage) {
        c3_fill_a<Elem, BK, BM, S::threads, TRANS>(
            p, C3PanelA<S, Elem>{sAbase + stage * S::kAMax}, rows, int64_t(kt) * BK, in, tid, K);
        c3_fill_b<Elem, BK, BN, S::threads>(p, C3PanelB<S, Elem>{sBbase + stage * S::kBMax}, n0,
                                            int64_t(kt) * BK, W, tid, K);
        cp_async_commit_group();
    };

    float acc[S::MA][S::NA][4];
#pragma unroll
    for (int i = 0; i < S::MA; ++i) {
#pragma unroll
        for (int j = 0; j < S::NA; ++j) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                acc[i][j][t] = 0.0f;
            }
        }
    }

#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(t, t % ST);
    }
    for (int kt = 0; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);
        f16_gemm_tile<S, Elem, false, false>(sAbase + stage * S::kAMax,
                                             sBbase + stage * S::kBMax, warp_m, warp_n, lane, acc);
    }

    // C: lane l holds (m = l/4, n = 2*(l%4)), plus the +1 column and the +8 row.
    const int cm = lane / 4;
    const int cn = 2 * (lane % 4);
#pragma unroll
    for (int i = 0; i < S::MA; ++i) {
        const int gm = warp_m * S::WM + i * 16 + cm;
#pragma unroll
        for (int j = 0; j < S::NA; ++j) {
            const int gn = n0 + warp_n * S::WN + j * 8 + cn;
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int row = gm + rr * 8;
                const int m = m0 + row;
                if (m >= p.m) {
                    continue;
                }
                const int64_t base = rows.obase[row];
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int n = gn + cc;
                    if (n >= p.ocg) {
                        continue;
                    }
                    float v = acc[i][j][rr * 2 + cc];
                    Elem* dst = out + base + int64_t(n) * p.out_c;
                    if (bias != nullptr) {
                        v += s_load(bias + n);
                    }
                    if (p.add_to && p.pre_act) {
                        v += s_load(dst);
                    }
                    v = fminf(fmaxf(v, p.min_clip), p.max_clip);
                    v = gemm_apply_activation(p.ep, p.ep_alpha, p.ep_beta, v);
                    if (p.add_to && !p.pre_act) {
                        v += s_load(dst);
                    }
                    s_store(dst, v);
                }
            }
        }
    }
}

template <class Elem, bool TRANS>
inline Status c3_launch(const C3Params& p, dim3 grid, cudaStream_t stream)
{
    using S = C3Shape<Elem, TRANS>;
    if constexpr (sizeof(Elem) == 4) {
        constexpr int smem = S::panel_bytes(false, false);
        conv3d_kernel_f<S, TRANS><<<grid, S::threads, smem, stream>>>(p);
    } else {
        constexpr int smem = S::kSmemBytes;
        conv3d_kernel_h<S, Elem, TRANS><<<grid, S::threads, smem, stream>>>(p);
    }
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda::conv3d
