/// @file conv2d_i8.cu
/// @brief CUDA Conv2D — the quantized route.
///
/// A pre-quantized u8/s8 activation against an s8 HWIO filter, MatMulInteger
/// shaped: the 16x8x32 s8 MMA over a 64-pixel by 64-channel CTA tile (128
/// channels where the group has them), the per-pixel activation sums folded in
/// during the gather, and the zero-point compensation applied once in the
/// epilogue. The activation carries its per-tensor parameters; the filter's
/// per-channel arrays are staged by the host entry, and its per-output-channel
/// sums (R_b) are either reduced on the device or read out of a packed filter.
///
/// The route also owns its prepacked filter — PackedHWIOCuda, the k-major
/// panels cut at the tile's own geometry plus the sums table (conv2d_pack.h) —
/// which turns the fill's sixteen strided byte gathers into one `cp.async`.
///
/// What not to "fix" later:
///
///  * The A fill is this kernel's dominant cost and its traffic is M·N·BK/BN, so
///    the 128-channel tile exists to halve it; the fill is 16 B a thread out of
///    a dense channel axis, and the int8 panel's row is the shared-memory row.
///  * The split-K cut takes eight k tiles a split, not four: at four the
///    two-tile pipeline prologue is half the work, and 28x28x128 k3 measured
///    0.88x.
///  * The epilogue stages its 8x sector scatter through a shared tile, and the
///    A fill's two-way store conflict is broken by swapping row bits 0 and 2
///    (conv_fill_row's Groups == 4 case) — both are why the int8 tile is not a
///    template clone of the fp one.
///  * The blob's static_assert pins its panel to the kernel's shared-memory
///    panel: the pack pass and the fill cannot drift apart.

#include "conv2d_common.cuh"
#include "conv2d_pack.h"
#include "conv2d_i8.cuh"
#include "../matmul/matmul_i8_common.cuh"
#include "../matmul/matmul_i8_ptx.cuh" // I8TileShape + the PTX helpers

namespace nnops::backend::cuda {
namespace {

// ============================================================
// int8 tensor-core kernel
// ============================================================

/// The int8 tile: 64 pixels by 64 channels, 64 bytes of k, three stages, and the
/// same eight-warp split as the f16 tile's [[ConvShapeH]] (32x16 per warp) rather
/// than four warps of 32x32. The s8 panels are half the fp16 footprint, so a
/// third stage still leaves three blocks a SM and 24 warps.
using ConvShapeI8 = I8TileShape<32, 16, 2, 4, 64, 3, 3>;

/// The same tile with 128 channels a CTA, for a group that has them. The A fill
/// is the int8 kernel's dominant cost and its total traffic is M*N*BK/BN, so
/// doubling BN halves it; the warp keeps a square-ish 32x32 fragment, which also
/// takes the A `ldmatrix` from one m16n8k32 operand set (LDSM:IMMA 1:1) to two.
using ConvShapeI8Wide = I8TileShape<32, 32, 2, 4, 64, 3, 2>;

/// The epilogue staging tile's row pitch, in bytes: the smallest pitch at least
/// `BN * esz` that is 80 mod 128. Four banks a row (80 bytes is 20 words) is
/// what the staging pass needs -- eight consecutive rows then land on eight
/// distinct banks -- and the copy-out reads 16 B at a time, where the pitch
/// only has to be a multiple of 16.
template <class S>
constexpr int conv_i8_stage_pitch(int esz)
{
    const int need = S::BN * esz;
    return need + (80 - need % 128 + 128) % 128;
}

static_assert(ConvShapeI8::BM * conv_i8_stage_pitch<ConvShapeI8>(4) <= ConvShapeI8::kSmemBytes,
              "the staging tile has to fit the pipeline's shared memory");
static_assert(ConvShapeI8Wide::BM * conv_i8_stage_pitch<ConvShapeI8Wide>(4) <=
                  ConvShapeI8Wide::kSmemBytes,
              "the staging tile has to fit the pipeline's shared memory");

/// Either k-major s8 panel, as the gather's (row, col, stage) indexer. The base
/// already carries the stage; rows are `LD` bytes apart.
template <class S, class Elem>
struct ConvI8Panel {
    Elem* base;
    __device__ __forceinline__ Elem& operator()(int row, int col, int) const
    {
        return base[row * S::LD + col];
    }
};

/// The int8 gather. Each thread owns one output pixel's
/// tile row, so the same walk returns that thread's share of R_a[pixel]; the
/// pixel a padded or out-of-range tap contributes is the real value 0, which is
/// also what it adds to the sum.
template <class Elem, int Threads, int bM, int bK, class SmemA>
__device__ inline int32_t conv_fill_a_i8(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                         int pixel0, int k0)
{
    const int M = p.oh * p.ow;
    const int K = p.icg * p.kh * p.kw;
    const int tid = int(threadIdx.x);
    const int inc = int(p.in_c), inh = int(p.in_h), inw = int(p.in_w);

    constexpr int per = Threads / bM; // k columns per thread
    const int row = tid % bM;
    const int pixel = pixel0 + row;
    const bool m_ok = pixel < M;
    const int oh = pixel / p.ow;
    const int ow = pixel - oh * p.ow;
    const int col0 = (tid / bM) * (bK / per);
    int k = k0 + col0;
    // k = tap*IC/G + ic: the channels are innermost inside one tap.
    const int tap = k / p.icg;
    int ic = k - tap * p.icg;
    int kh = tap / p.kw;
    int kw = tap - kh * p.kw;

    int32_t sum = 0;
    for (int j = 0; j < bK / per; ++j) {
        const int ih = oh * p.sh - p.ph + kh * p.dh;
        const int iw = ow * p.sw - p.pw + kw * p.dw;
        // A window-padded tap is the real value 0, stored as the activation's
        // zero point -- what the CPU im2col does. Filling 0 instead would leave
        // -(zp_x)*(qw - zp_w) per padded tap in the compensated sum.
        Elem val = Elem(0);
        if (m_ok && k < K) {
            val = (unsigned(ih) < unsigned(p.ih) && unsigned(iw) < unsigned(p.iw)) ?
                      in[ic * inc + ih * inh + iw * inw] :
                      Elem(p.zp_x);
        }
        sA(row, col0 + j, stage) = val;
        sum += int32_t(val);
        ++k;
        if (++ic == p.icg) {
            ic = 0;
            if (++kw == p.kw) {
                kw = 0;
                ++kh;
            }
        }
    }
    return sum;
}

/// The int8 vector fill, one 16 B vector per (row, k-group): 16 consecutive k are
/// 16 consecutive channels inside one tap, contiguous in the activation and in
/// the shared tile. A thread owns `rows.count` pixels, so it keeps one partial
/// per owned row and folds them into shared memory once, after the k loop.
template <class Elem, int bK, class SmemA>
__device__ inline void conv_fill_a_vec_i8(const ConvParams& p, SmemA sA, int stage, const Elem* in,
                                          const ConvRows& rows, int k0, ConvTapWalk& walk,
                                          int32_t* sums, bool add)
{
    constexpr int kGroup = 16 / int(sizeof(Elem));
    using Vec = uint4;

    const int K = p.icg * p.kh * p.kw;
    const int kg = int(threadIdx.x) % (bK / kGroup);
    const int k = k0 + kg * kGroup;
    conv_tap_seek(p, walk, k);
    const int ic = walk.ic, kh = walk.kh, kw = walk.kw;
    const int inc = int(p.in_c), inh = int(p.in_h), inw = int(p.in_w);
    const int koff = ic * inc + (kh * p.dh) * inh + (kw * p.dw) * inw;
    const int dh = kh * p.dh, dw = kw * p.dw;
    const bool k_ok = k < K;
    const Vec zero = {};
    // A padded tap carries the activation's zero point, so the compensation
    // cancels it exactly (see conv_fill_a_i8).
    Vec zpv = zero;
    {
        Elem* e = reinterpret_cast<Elem*>(&zpv);
    #pragma unroll
        for (int t = 0; t < kGroup; ++t) {
            e[t] = Elem(p.zp_x);
        }
    }

#pragma unroll
    for (int i = 0; i < ConvRows::kMax; ++i) {
        if (i >= rows.count) {
            break;
        }
        Vec v = zero;
        if (k_ok && rows.ok[i]) {
            const int ih = rows.oh0[i] + dh;
            const int iw = rows.ow0[i] + dw;
            v = (unsigned(ih) < unsigned(p.ih) && unsigned(iw) < unsigned(p.iw)) ?
                    *reinterpret_cast<const Vec*>(in + koff + rows.off[i]) :
                    zpv;
        }
        *reinterpret_cast<Vec*>(&sA(rows.row[i], kg * kGroup, stage)) = v;
        if (add) {
            // Four bytes a step: the scalar fold was a shift and an add per
            // element -- 32 of the fill's ~143 instructions per k tile.
            const uint32_t* const w = reinterpret_cast<const uint32_t*>(&v);
            int32_t sum = 0;
        #pragma unroll
            for (int t = 0; t < kGroup / 4; ++t) {
                if constexpr (std::is_same_v<Elem, uint8_t>) {
                    sum = int(__dp4a(w[t], 0x01010101u, unsigned(sum)));
                } else {
                    sum = __dp4a(int(w[t]), 0x01010101, sum);
                }
            }
            sums[i] += sum;
        }
    }
}

/// R_b[oc] = Σ_k filter[oc,k] over the whole reduction, per global channel.
__global__ void conv_i8_col_sums_kernel(ConvParams p)
{
    const int oc = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (oc >= p.oc) {
        return;
    }
    const int g = oc / p.ocg;
    const int ocl = oc - g * p.ocg;
    const int8_t* w =
        static_cast<const int8_t*>(p.weight) + int64_t(g) * p.ocg * p.w_n + int64_t(ocl) * p.w_n;
    int32_t s = 0;
    for (int ic = 0; ic < p.icg; ++ic) {
        for (int kh = 0; kh < p.kh; ++kh) {
            for (int kw = 0; kw < p.kw; ++kw) {
                s += int32_t(w[ic * p.w_ic + kh * p.w_kh + kw * p.w_kw]);
            }
        }
    }
    p.r_b[oc] = s;
}

// ============================================================
// int8 packed filter (PackedHWIOCuda)
// ============================================================

/// The blob's geometry, derived from the filter's shape and the attributes so
/// the descriptor query, the pack pass and the kernel cannot disagree. Bytes:
/// `[group][k tile][oc block]` panels of BN rows at LD bytes, then the
/// per-channel sums as `[group][OC/G]` int32.
struct ConvI8PackGeom {
    bool ok = false;
    int groups = 0, icg = 0, ocg = 0, oc = 0;
    int kh = 0, kw = 0;
    int K = 0, ktiles = 0, nblk = 0;
    int64_t panel = 0;    ///< bytes of one panel
    int64_t data = 0;     ///< bytes of every panel
    int64_t sums_off = 0; ///< where the int32 sums start
    int64_t total = 0;
};

inline ConvI8PackGeom conv_i8_pack_geom(int64_t groups, int64_t icg, int64_t ocg, int64_t kh,
                                        int64_t kw)
{
    ConvI8PackGeom g;
    if (groups <= 0 || icg <= 0 || ocg <= 0 || kh <= 0 || kw <= 0) {
        return g;
    }
    const int64_t K = icg * kh * kw;
    const int64_t ktiles = (K + kConvI8PackBK - 1) / kConvI8PackBK;
    const int64_t nblk = (ocg + kConvI8PackBN - 1) / kConvI8PackBN;
    const int64_t oc = groups * ocg;
    g.groups = int(groups);
    g.icg = int(icg);
    g.ocg = int(ocg);
    g.oc = int(oc);
    g.kh = int(kh);
    g.kw = int(kw);
    g.K = int(K);
    g.ktiles = int(ktiles);
    g.nblk = int(nblk);
    g.panel = int64_t(kConvI8PackBN) * kConvI8PackLD;
    g.data = groups * ktiles * nblk * g.panel;
    g.sums_off = (g.data + 15) & ~int64_t(15);
    g.total = g.sums_off + oc * 4;
    g.ok = groups * ocg <= 65535 && K <= (int64_t(1) << 24) &&
           g.total <= (int64_t(1) << 40);
    return g;
}

struct ConvI8PackParams {
    const void* weight;
    void* blob;
    ConvI8PackGeom g;
    int32_t w_n, w_ic, w_kh, w_kw;
};

/// One launch builds the whole blob: the panels first, then the sums. The
/// gather is the one the kernel used to run once per block per k tile, so
/// paying it here once per call — or never, when the caller caches the blob —
/// is the entire point. Byte-wise, because that is what a pack pass can afford
/// and the kernel's fill could not.
__global__ void conv_i8_pack_kernel(ConvI8PackParams q)
{
    constexpr int BN = kConvI8PackBN, BK = kConvI8PackBK, LD = kConvI8PackLD;
    constexpr int kVec = BK / 16;
    const int64_t per_panel = int64_t(BN) * kVec;
    const int G = q.g.groups, K = q.g.K, kt = q.g.ktiles, nb = q.g.nblk;
    const int64_t chunks = int64_t(G) * kt * nb * per_panel;
    int8_t* const blob = static_cast<int8_t*>(q.blob);
    const int8_t* const W = static_cast<const int8_t*>(q.weight);
    const int32_t w_n = q.w_n, w_ic = q.w_ic, w_kh = q.w_kh, w_kw = q.w_kw;
    const int32_t icg = q.g.icg, kw_end = q.g.kw;

    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < chunks;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t panel = i / per_panel;
        const int64_t r = i - panel * per_panel;
        const int row = int(r / kVec);
        const int c = int(r - int64_t(row) * kVec);
        const int nbi = int(panel % nb);
        const int kti = int((panel / nb) % kt);
        const int g = int(panel / (int64_t(nb) * kt));
        const int n = nbi * BN + row;
        const int k0 = kti * BK + c * 16;
        int8_t* const d = blob + panel * q.g.panel + row * LD + c * 16;
        if (n >= q.g.ocg) {
            *reinterpret_cast<uint4*>(d) = uint4{};
            continue;
        }
        const int8_t* const row_w = W + int64_t(g) * q.g.ocg * w_n + int64_t(n) * w_n;
        const int32_t tap = k0 / icg;
        int32_t ic = k0 - tap * icg;
        int32_t kh = tap / kw_end;
        int32_t kw = tap - kh * kw_end;
        int32_t off = ic * w_ic + kh * w_kh + kw * w_kw;
        int k = k0;
        uint4 v;
        int8_t* const vb = reinterpret_cast<int8_t*>(&v);
#pragma unroll 4
        for (int j = 0; j < 16; ++j) {
            vb[j] = k < K ? row_w[off] : int8_t(0);
            ++k;
            off += w_ic;
            if (++ic == icg) {
                ic = 0;
                off -= icg * w_ic;
                off += w_kw;
                if (++kw == kw_end) {
                    kw = 0;
                    off -= kw_end * w_kw;
                    off += w_kh;
                }
            }
        }
        *reinterpret_cast<uint4*>(d) = v;
    }

    int32_t* const sums = reinterpret_cast<int32_t*>(blob + q.g.sums_off);
    for (int oc = int(blockIdx.x) * blockDim.x + threadIdx.x; oc < q.g.oc;
         oc += int(gridDim.x) * blockDim.x) {
        const int8_t* const row_w = W + int64_t(oc) * w_n;
        int32_t s = 0;
        for (int32_t kh = 0; kh < q.g.kh; ++kh) {
            for (int32_t kw = 0; kw < q.g.kw; ++kw) {
                for (int32_t ic = 0; ic < icg; ++ic) {
                    s += int32_t(row_w[ic * w_ic + kh * w_kh + kw * w_kw]);
                }
            }
        }
        sums[oc] = s;
    }
}

template <class S, class AElem>
__global__ __launch_bounds__(S::threads, S::MinBlocks)
void conv_gemm_i8_kernel(ConvParams p)
{
    using BElem = int8_t;
    constexpr int BM = S::BM;
    constexpr int BN = S::BN;
    constexpr int BK = S::BK;
    constexpr int ST = S::stages;
    constexpr int LD = S::LD;

    extern __shared__ __align__(16) unsigned char conv_smem[];
    int8_t* const sAbase = reinterpret_cast<int8_t*>(conv_smem);
    int8_t* const sBbase = sAbase + ST * S::kABytes;

    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int K = p.icg * p.kh * p.kw;
    const int m0 = int(blockIdx.x) * BM;
    const int n0 = int(blockIdx.y) * BN;
    const int tid = int(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::wn;
    const int warp_n = warp - warp_m * S::wn;

    const ConvBlock blk = conv_block(p);
    const int zb = blk.zb;
    const int g = blk.g;
    const int gi = g * p.ocg;

    const int64_t in_base = int64_t(zb) * p.in_batch + int64_t(g) * p.icg * p.in_c;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;

    BElem const* W = static_cast<BElem const*>(p.weight) + int64_t(g) * p.ocg * p.w_n;
    AElem const* in = static_cast<AElem const*>(p.input) + in_base;

    // The per-pixel activation sums, folded into shared memory after the k
    // loop. The scalar fill gives one partial per thread (each owns one row);
    // the vector fill owns several rows, so it keeps an array.
    __shared__ int32_t s_ra[BM];
    if (tid < BM) {
        s_ra[tid] = 0;
    }
    __syncthreads();
    const ConvRows rows = conv_rows<AElem, S::threads, BM, BK>(p, m0);
    int32_t ra_single = 0;
    int32_t ra_vec[ConvRows::kMax] = {};
    int ra_kt = -1;
    // The fill's k-region odometer, as on the fp tiles: the ring asks for
    // monotonic k, so the tap decode is carried instead of divided.
    ConvTapWalk tap_walk;

    // The blob's own k tiling, then the slice of it this block owns: a split is
    // a proportional cut, and the sums it hands the reduce carry their own k
    // count -- the only term that cannot be summed later.
    const int k_tiles = (K + BK - 1) / BK;
    int kt_begin = 0, kt_count = 0;
    conv_split_range(K, BK, p.k_splits, blk.split, kt_begin, kt_count);
    const int kt_end = kt_begin + kt_count;
    const int K_split = std::min(K, kt_end * BK) - kt_begin * BK;
    // The filter gather's bytes per thread per k tile, and whether such a run
    // can cross a tap. It cannot when icg is a multiple of the run, which the
    // tail check below is what makes sufficient.
    constexpr int kBFillK = BK / (S::threads / BN);
    constexpr int kNVec = kBFillK / 16;
    static_assert(kBFillK % 16 == 0, "the filter gather writes whole 16 B vectors");
    // A packed-filter panel is kConvI8PackBN rows of the format's own stride, so
    // a tile wider than the blob reads one panel per block of channels.
    constexpr int kNPanels = BN / kConvI8PackBN;
    constexpr int kPanelBytes = kConvI8PackBN * kConvI8PackLD;
    static_assert(BN % kConvI8PackBN == 0, "a tile has to read whole blob panels");
    const bool w_tap_aligned = (p.icg % kBFillK) == 0;
    ConvTapWalk w_walk;

    auto fill = [&](int kt, int stage) {
        if (kt >= kt_end) {
            cp_async_commit_group();
            return;
        }
        // A k tile is gathered once; the guard keeps the fold that way if the
        // ring is ever allowed to re-issue a tile.
        const bool add = (kt != ra_kt);
        if (add) {
            ra_kt = kt;
        }
        const int k0 = kt * BK;
        int8_t* const sA = sAbase + stage * S::kABytes;
        int8_t* const sB = sBbase + stage * S::kBBytes;
        if (p.fill_vec) {
            conv_fill_a_vec_i8<AElem, BK>(p, ConvI8Panel<S, AElem>{reinterpret_cast<AElem*>(sA)},
                                          stage, in, rows, k0, tap_walk, ra_vec, add);
        } else {
            const int32_t s =
                conv_fill_a_i8<AElem, S::threads, BM, BK>(
                    p, ConvI8Panel<S, AElem>{reinterpret_cast<AElem*>(sA)}, stage, in, m0, k0);
            if (add) {
                ra_single += s;
            }
        }
        // The filter is HWIO with the channel innermost, so its k axis is not
        // contiguous in memory and B is gathered byte-wise into a k-major panel.
        // The loads are the same either way; the store is what the tile pays
        // for. Thirty-two lanes storing one byte each sit on thirty-two rows at
        // a 16 B stride, which reaches two of the eight bank quads: 4 wavefronts
        // a byte, so the sixteen a thread owns cost sixty-four where one 16 B
        // vector costs four. ncu on 448x448x64: 12.4M of the kernel's 16.9M
        // store wavefronts were conflicts, against 79,799 on all the ldmatrix
        // reads together.
        //
        // A prepacked filter removes the gather altogether — and the sixteen
        // strided loads with it, which the vector store above leaves alone.
        {
            const int row = tid % BN;
            const int col0 = (tid / BN) * kBFillK;
            int8_t* const d = sB + row * LD + col0;
            const int n = n0 + row;
            const int8_t* const row_w = W + int64_t(n) * p.w_n;
            const int k = k0 + col0;
            const bool n_ok = n < N;
            const int32_t w_ic = int32_t(p.w_ic), w_kh = int32_t(p.w_kh), w_kw = int32_t(p.w_kw);
            if (p.w_panel != nullptr) {
                // The blob already holds this panel row for row: no predicate
                // (its tails are zeroed), no odometer, one asynchronous copy.
                // A blob panel is kConvI8PackBN rows, so a tile wider than that
                // reads one panel per block of channels; a block past the last
                // panel is left alone, and the epilogue skips those channels.
                const int half = row / kConvI8PackBN;
                const int64_t pan = int64_t(blockIdx.y) * kNPanels + half;
                if (pan < p.w_nblk) {
                    const int8_t* const src = static_cast<const int8_t*>(p.w_panel) +
                                              ((int64_t(g) * k_tiles + kt) * p.w_nblk + pan) *
                                                  kPanelBytes +
                                              (row - half * kConvI8PackBN) * LD + col0;
#pragma unroll
                    for (int c = 0; c < kNVec; ++c) {
                        cp_async_ca_16B(d + c * 16, src + c * 16);
                    }
                }
            } else if (n_ok && k + kBFillK <= K && w_tap_aligned) {
                // Inside one tap the run is a fixed `w_ic` stride, so the
                // odometer the fallback steps is not needed at all.
                conv_tap_seek(p, w_walk, k);
                const int32_t off0 = w_walk.ic * w_ic + w_walk.kh * w_kh + w_walk.kw * w_kw;
#pragma unroll
                for (int c = 0; c < kNVec; ++c) {
                    uint4 v;
                    int8_t* const vb = reinterpret_cast<int8_t*>(&v);
                    const int32_t off = off0 + c * 16 * w_ic;
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        vb[j] = row_w[off + j * w_ic];
                    }
                    *reinterpret_cast<uint4*>(d + c * 16) = v;
                }
            } else {
                const int32_t icg = int32_t(p.icg), kw_end = int32_t(p.kw);
                int kk = k;
                const int32_t tap0 = kk / icg;
                int32_t ic = kk - tap0 * icg;
                int32_t kh = tap0 / kw_end;
                int32_t kw = tap0 - kh * kw_end;
                int32_t off = ic * w_ic + kh * w_kh + kw * w_kw;
#pragma unroll 4
                for (int j = 0; j < kBFillK; ++j) {
                    d[j] = (n_ok && kk < K) ? row_w[off] : BElem(0);
                    ++kk;
                    off += w_ic;
                    if (++ic == icg) {
                        ic = 0;
                        off -= icg * w_ic;
                        off += w_kw;
                        if (++kw == kw_end) {
                            kw = 0;
                            off -= kw_end * w_kw;
                            off += w_kh;
                        }
                    }
                }
            }
        }
        cp_async_commit_group();
    };

    int32_t acc[S::MA][S::NA][4];
#pragma unroll
    for (int i = 0; i < S::MA; ++i)
#pragma unroll
        for (int j = 0; j < S::NA; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                acc[i][j][t] = 0;
            }

#pragma unroll
    for (int t = 0; t < ST - 1; ++t) {
        fill(kt_begin + t, (kt_begin + t) % ST);
    }
    for (int kt = kt_begin; kt < kt_end; ++kt) {
        const int stage = kt % ST;
        // One barrier a tile: the wait retires this tile's group, the barrier
        // then publishes the gather's plain stores and retires every warp's
        // reads of the tile before it -- whose stage the fill now overwrites.
        cp_async_wait_group<ST - 2>();
        __syncthreads();
        fill(kt + ST - 1, (kt + ST - 1) % ST);
        i8_gemm_tile<S, AElem>(sAbase + stage * S::kABytes, sBbase + stage * S::kBBytes,
                                      warp_m, warp_n, lane, acc);
    }

    if (p.fill_vec) {
        for (int i = 0; i < rows.count; ++i) {
            atomicAdd(&s_ra[rows.row[i]], ra_vec[i]);
        }
    } else {
        atomicAdd(&s_ra[tid % BM], ra_single);
    }
    __syncthreads();

    // ---- output ---------------------------------------------------------
    const I8OutKind kind = static_cast<I8OutKind>(p.out_kind);
    const bool int_out = kind == I8OutKind::S32 || kind == I8OutKind::S8 || kind == I8OutKind::U8;
    const int32_t* bias32 =
        int_out && p.bias != nullptr ? static_cast<const int32_t*>(p.bias) + gi : nullptr;
    const float* biasf32 = kind == I8OutKind::F32 && p.bias != nullptr ?
                               static_cast<const float*>(p.bias) + gi :
                               nullptr;
    const __half* biasf16 = kind == I8OutKind::F16 && p.bias != nullptr ?
                                static_cast<const __half*>(p.bias) + gi :
                                nullptr;

    // A lane's fragment carries the cc pair of one pixel's channel run, so a
    // direct store moves four useful bytes per sector it touches -- eight rows
    // an instruction. ncu on 448x448x64 read 3.21M store sectors for a tile
    // whose bytes need 0.40M. The tile is staged in the pipeline's shared
    // memory instead (dead behind the barrier above) and copied out as whole
    // 16 B vectors, which is a sector's worth of one pixel's row.
    const int esz = kind == I8OutKind::F32 ? 4 : (kind == I8OutKind::F16 ? 2 : 1);
    const int pitch = conv_i8_stage_pitch<S>(esz);
    unsigned char* const sC = conv_smem;

    // A split's block stops at its accumulator plane; the whole-K compensation,
    // the bias and the requant all belong to the reduce.
    int32_t* const part =
        p.k_splits > 1 ? p.partial_i32 + int64_t(blockIdx.z) * int64_t(M) * N : nullptr;

    // C: lane l holds (m = l/4, n = 2*(l%4)), plus the +1 column and the +8 row.
    const int cm = lane / 4;
    const int cn = 2 * (lane % 4);
#pragma unroll
    for (int i = 0; i < S::MA; ++i) {
        const int gm = m0 + warp_m * S::WM + i * 16 + cm;
#pragma unroll
        for (int j = 0; j < S::NA; ++j) {
            const int gn = n0 + warp_n * S::WN + j * 8 + cn;
            const int cs = warp_n * S::WN + j * 8 + cn;
#pragma unroll
            for (int rr = 0; rr < 2; ++rr) {
                const int m = gm + rr * 8;
                if (m >= M) {
                    continue;
                }
                unsigned char* const srow = sC + (m - m0) * pitch;
#pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int n = gn + cc;
                    if (n >= N) {
                        continue;
                    }
                    const int32_t zpw = p.zp_w != nullptr ? p.zp_w[gi + n] : p.zp_w_scalar;
                    int32_t v = acc[i][j][rr * 2 + cc];
                    if (part != nullptr) {
                        // A split stops at its own slice: the activation term
                        // uses its own k count and the weight term is left to
                        // the reduce, which is the only place the whole K is
                        // known. `m` is the global pixel; sums are tile rows.
                        v -= (s_ra[m - m0] - K_split * p.zp_x) * zpw;
                        part[int64_t(m) * N + n] = v;
                        continue;
                    }
                    v -= p.zp_x * p.r_b[gi + n];
                    // `m` is the global pixel; the sums are indexed by tile row.
                    v -= (s_ra[m - m0] - K * p.zp_x) * zpw;

                    switch (kind) {
                    case I8OutKind::S32:
                        // Refused by the entry: the raw int32 destination is
                        // MatMul's.
                        break;
                    case I8OutKind::S8:
                    case I8OutKind::U8: {
                        if (bias32 != nullptr) {
                            v += bias32[n];
                        }
                        if (p.relu) {
                            v = v < 0 ? 0 : v;
                        }
                        const float s_w = p.sc_w != nullptr ? p.sc_w[gi + n] : p.sc_w_scalar;
                        const float row_k = s_w / p.out_scale;
                        if (kind == I8OutKind::S8) {
                            reinterpret_cast<int8_t*>(srow)[cs + cc] = detail::i8_requant<int8_t>(
                                v, row_k, p.sc_x, float(p.out_zp));
                        } else {
                            reinterpret_cast<uint8_t*>(srow)[cs + cc] = detail::i8_requant<uint8_t>(
                                v, row_k, p.sc_x, float(p.out_zp));
                        }
                        break;
                    }
                    case I8OutKind::F32:
                    case I8OutKind::F16: {
                        const float s_w = p.sc_w != nullptr ? p.sc_w[gi + n] : p.sc_w_scalar;
                        float f = float(v) * (s_w * p.sc_x);
                        float b = 0.0f;
                        if (biasf32 != nullptr) {
                            b = biasf32[n];
                        } else if (biasf16 != nullptr) {
                            b = __half2float(biasf16[n]);
                        }
                        if (b != 0.0f) {
                            f += b;
                        }
                        f = fminf(fmaxf(f, p.min_clip), p.max_clip);
                        f = gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, f);
                        if (kind == I8OutKind::F32) {
                            reinterpret_cast<float*>(srow)[cs + cc] = f;
                        } else {
                            reinterpret_cast<__half*>(srow)[cs + cc] = __float2half(f);
                        }
                        break;
                    }
                    }
                }
            }
        }
    }
    __syncthreads();

    // ---- copy-out --------------------------------------------------------
    // Phase two moves bytes: the staged tile already holds the destination's
    // bytes, row-major by channel. The widest run a row allows is the unit: 16
    // elements of s8 need the pixel pitch to be a multiple of 16, and a channel
    // count like 24 drops to a 4 B unit instead of falling all the way back.
    const auto fits = [&](int u) {
        return p.out_c == 1 && (p.out_w % u) == 0 && (p.out_h % u) == 0;
    };
    const int unit = fits(16 / esz) ? 16 / esz : (fits(4 / esz) ? 4 / esz : 1);
    const int chunks = S::BN / unit;
    const int width = unit * esz;
    unsigned char* const dst = static_cast<unsigned char*>(p.output);
    for (int v = tid; v < S::BM * chunks; v += S::threads) {
        const int row = v / chunks;
        const int ch = (v - row * chunks) * unit;
        const int m = m0 + row;
        if (m >= M || n0 + ch >= N) {
            continue;
        }
        const int oh = m / p.ow;
        const int ow = m - oh * p.ow;
        const int64_t px = out_base + int64_t(oh) * p.out_h + int64_t(ow) * p.out_w;
        const unsigned char* const src = sC + row * pitch + ch * esz;
        unsigned char* const d = dst + (px + n0 + ch) * esz;
        const uintptr_t a = reinterpret_cast<uintptr_t>(d);
        const bool vec = n0 + ch + unit <= N &&
                         (width < 4 || (a & (width == 16 ? 15u : 3u)) == 0);
        if (vec) {
            if (width == 16) {
                *reinterpret_cast<uint4*>(d) = *reinterpret_cast<const uint4*>(src);
            } else if (width == 4) {
                *reinterpret_cast<uint32_t*>(d) = *reinterpret_cast<const uint32_t*>(src);
            } else {
                for (int b = 0; b < width; ++b) {
                    d[b] = src[b];
                }
            }
            continue;
        }
        for (int c = 0; c < unit; ++c) {
            const int n = n0 + ch + c;
            if (n >= N) {
                break;
            }
            const unsigned char* const s = src + c * esz;
            unsigned char* const o = dst + (px + int64_t(n) * p.out_c) * esz;
            for (int b = 0; b < esz; ++b) {
                o[b] = s[b];
            }
        }
    }
}

/// The int8 epilogue's bias and relu, on the compensated accumulator. Shared by
/// the tile and by the split reduce so the two cannot drift.
__device__ inline int32_t conv_i8_out_int(const ConvParams& p, int n, int32_t v,
                                          const int32_t* bias32)
{
    if (bias32 != nullptr) {
        v += bias32[n];
    }
    if (p.relu) {
        v = v < 0 ? 0 : v;
    }
    return v;
}

/// The same for a dequantizing destination: scale, bias, clip, activation.
__device__ inline float conv_i8_out_f32(const ConvParams& p, int gi, int n, int32_t v,
                                        const float* biasf32, const __half* biasf16)
{
    const float s_w = p.sc_w != nullptr ? p.sc_w[gi + n] : p.sc_w_scalar;
    float f = float(v) * (s_w * p.sc_x);
    float b = 0.0f;
    if (biasf32 != nullptr) {
        b = biasf32[n];
    } else if (biasf16 != nullptr) {
        b = __half2float(biasf16[n]);
    }
    if (b != 0.0f) {
        f += b;
    }
    f = fminf(fmaxf(f, p.min_clip), p.max_clip);
    return gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, f);
}

/// The int8 split's second pass: sum the planes, apply the whole-K weight
/// compensation no split could see, then the epilogue once -- the same order the
/// single-pass kernel runs. One thread per output element, so its stores are
/// coalesced whatever the fragment order was.
__global__ void conv_i8_splitk_reduce_kernel(ConvParams p)
{
    const I8OutKind kind = static_cast<I8OutKind>(p.out_kind);
    const int M = p.oh * p.ow;
    const int N = p.ocg;
    const int64_t plane = int64_t(M) * N;
    const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= plane) {
        return;
    }
    const int pixel = int(idx / N);
    const int n = int(idx - int64_t(pixel) * N);
    const int zg = int(blockIdx.y);

    const int32_t* src = p.partial_i32 + int64_t(zg) * p.k_splits * plane + idx;
    int32_t v = 0;
    for (int s = 0; s < p.k_splits; ++s) {
        v += src[int64_t(s) * plane];
    }

    const int zb = zg / p.groups;
    const int g = zg - zb * p.groups;
    const int gi = g * p.ocg;
    const int64_t out_base = int64_t(zb) * p.out_batch + int64_t(g) * p.ocg * p.out_c;
    v -= p.zp_x * p.r_b[gi + n];

    switch (kind) {
    case I8OutKind::S32:
        break;
    case I8OutKind::S8:
    case I8OutKind::U8: {
        const int32_t* bias32 =
            p.bias != nullptr ? static_cast<const int32_t*>(p.bias) + gi : nullptr;
        const int32_t q = conv_i8_out_int(p, n, v, bias32);
        const float s_w = p.sc_w != nullptr ? p.sc_w[gi + n] : p.sc_w_scalar;
        const float row_k = s_w / p.out_scale;
        if (kind == I8OutKind::S8) {
            *conv_out_addr<int8_t>(p, out_base, pixel, n) =
                detail::i8_requant<int8_t>(q, row_k, p.sc_x, float(p.out_zp));
        } else {
            *conv_out_addr<uint8_t>(p, out_base, pixel, n) =
                detail::i8_requant<uint8_t>(q, row_k, p.sc_x, float(p.out_zp));
        }
        break;
    }
    case I8OutKind::F32:
    case I8OutKind::F16: {
        const float* biasf32 = kind == I8OutKind::F32 && p.bias != nullptr ?
                                   static_cast<const float*>(p.bias) + gi :
                                   nullptr;
        const __half* biasf16 = kind == I8OutKind::F16 && p.bias != nullptr ?
                                    static_cast<const __half*>(p.bias) + gi :
                                    nullptr;
        const float f = conv_i8_out_f32(p, gi, n, v, biasf32, biasf16);
        if (kind == I8OutKind::F32) {
            *conv_out_addr<float>(p, out_base, pixel, n) = f;
        } else {
            *conv_out_addr<__half>(p, out_base, pixel, n) = __float2half(f);
        }
        break;
    }
    }
}

/// The int8 split reduce, kept beside its workspace the way the fp tiles do it.
inline void conv_i8_reduce(ConvParams& p, detail::GemmWorkspace ws, cudaStream_t stream)
{
    constexpr int kReduceThreads = 256;
    const int64_t plane = int64_t(p.oh) * p.ow * p.ocg;
    dim3 rgrid(unsigned((plane + kReduceThreads - 1) / kReduceThreads),
               unsigned(p.batch * p.groups), 1);
    conv_i8_splitk_reduce_kernel<<<rgrid, kReduceThreads, 0, stream>>>(p);
    detail::gemm_workspace_free(ws, stream);
}

template <class AElem> void launch_conv_i8(ConvParams& p, cudaStream_t stream)
{
    // A group narrower than the wide tile would waste half its MMA work, so the
    // 64-channel shape keeps the layers that have exactly that.
    if (p.ocg >= ConvShapeI8Wide::BN) {
        using S = ConvShapeI8Wide;
        constexpr int smem = S::kSmemBytes;
        auto kernel = conv_gemm_i8_kernel<S, AElem>;
        static const cudaError_t cfg =
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        (void)cfg;
        // Eight k tiles a split: the two-tile pipeline prologue is then at
        // most a quarter of it. At four the cut measured 0.88x on 28x28x128 k3.
        const detail::GemmWorkspace ws = conv_split_workspace(p, S::BM, S::BN, S::BK, smem, 4,
                                                              stream, 8);
        p.partial_i32 = reinterpret_cast<int32_t*>(p.partial);
        dim3 grid(unsigned((p.oh * p.ow + S::BM - 1) / S::BM),
                  unsigned((p.ocg + S::BN - 1) / S::BN),
                  unsigned(p.batch * p.groups * p.k_splits));
        kernel<<<grid, S::threads, smem, stream>>>(p);
        if (p.k_splits > 1) {
            conv_i8_reduce(p, ws, stream);
        }
        return;
    }
    using S = ConvShapeI8;
    constexpr int smem = S::kSmemBytes;
    auto kernel = conv_gemm_i8_kernel<S, AElem>;
    static const cudaError_t cfg =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    (void)cfg;

    const detail::GemmWorkspace ws = conv_split_workspace(p, S::BM, S::BN, S::BK, smem, 4,
                                                          stream, 8);
    p.partial_i32 = reinterpret_cast<int32_t*>(p.partial);
    dim3 grid(unsigned((p.oh * p.ow + S::BM - 1) / S::BM),
              unsigned((p.ocg + S::BN - 1) / S::BN),
              unsigned(p.batch * p.groups * p.k_splits));
    kernel<<<grid, S::threads, smem, stream>>>(p);
    if (p.k_splits > 1) {
        conv_i8_reduce(p, ws, stream);
    }
}

} // anonymous namespace

/// The quantized route's CUDA entry: pre-quantized u8/s8 activation against an
/// s8 HWIO filter, MatMulInteger-shaped output. The weight's
/// per-channel parameters are host arrays by contract and staged here, exactly
/// as the dense GEMM does.
Status conv2d_i8_cuda(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const TensorView& act = inputs[0];
    const TensorView& wgt = inputs[1];
    const TensorView* bias = inputs.size() > 2 ? &inputs[2] : nullptr;

    const DataType adt = act.data_type();
    if (adt != DataType::u8 && adt != DataType::s8) {
        return Status::unsupported;
    }
    if (wgt.data_type() != DataType::s8) {
        return Status::unsupported;
    }
    if (attrs.algorithm == ConvAlgorithm::Winograd) {
        return Status::unsupported;
    }
    // PackedHWIO and the SME blob are the CPU routes' products and the Winograd
    // U has no CUDA consumer, so only this route's own blob is admitted here.
    const bool w_packed = is_cuda_conv_weight(wgt.layout());
    if (is_packed_conv_weight(wgt.layout()) || is_wino_weight(wgt.layout()) ||
        (w_packed && wgt.rank() != 2)) {
        return Status::unsupported;
    }
    // The int8 store has no residual stage, so an add_to conv is refused.
    if (attrs.add_to) {
        return Status::unsupported;
    }

    I8OutKind kind;
    if (!i8_out_kind(output.data_type(), kind) || kind == I8OutKind::S32) {
        return Status::unsupported;
    }

    ConvParams p;
    p.batch = static_cast<int>(act.shape(0));
    p.ic = static_cast<int>(act.shape(3));
    p.ih = static_cast<int>(act.shape(1));
    p.iw = static_cast<int>(act.shape(2));
    p.oc = static_cast<int>(output.shape(3));
    p.oh = static_cast<int>(output.shape(1));
    p.ow = static_cast<int>(output.shape(2));
    p.groups = static_cast<int>(attrs.groups);
    // A blob's descriptor leads with OC, so the reduction's channel count comes
    // from the activation; a raw filter carries it on its third axis.
    p.icg = w_packed ? p.ic / p.groups : static_cast<int>(wgt.shape(2));
    p.ocg = p.oc / p.groups;
    p.kh = static_cast<int>(attrs.kernel_size[0]);
    p.kw = static_cast<int>(attrs.kernel_size[1]);
    p.sh = static_cast<int>(attrs.stride[0]);
    p.sw = static_cast<int>(attrs.stride[1]);
    p.dh = static_cast<int>(attrs.dilation[0]);
    p.dw = static_cast<int>(attrs.dilation[1]);
    p.ph = static_cast<int>(attrs.padding[0]);
    p.pw = static_cast<int>(attrs.padding[1]);

    p.in_batch = act.stride_elems(0);
    p.in_c = act.stride_elems(3);
    p.in_h = act.stride_elems(1);
    p.in_w = act.stride_elems(2);
    const int64_t esz = static_cast<int64_t>(data_type_size(output.data_type()));
    (void)esz;
    p.out_batch = output.stride_elems(0);
    p.out_c = output.stride_elems(3);
    p.out_h = output.stride_elems(1);
    p.out_w = output.stride_elems(2);

    p.input = act.ptr<void>();
    p.weight = wgt.ptr<void>();
    p.output = output.ptr<void>();
    p.bias = bias != nullptr ? bias->ptr<void>() : nullptr;

    p.epilogue = attrs.epilogue.type;
    p.ep_alpha = attrs.epilogue.alpha;
    p.ep_beta = attrs.epilogue.beta;
    p.min_clip = attrs.epilogue.min_clip;
    p.max_clip = attrs.epilogue.max_clip;
    p.add_to = false;

    // HWIO [KH][KW][IC/G][OC]: n is innermost, k is not contiguous. A blob has
    // no such axes -- its descriptor is [OC, ...] -- so the gather's strides are
    // only meaningful on the raw arm.
    if (!w_packed) {
        p.w_kh = wgt.stride_elems(0);
        p.w_kw = wgt.stride_elems(1);
        p.w_ic = wgt.stride_elems(2);
        p.w_n = wgt.stride_elems(3);
        p.w_k_dense = p.w_n == 1 && p.w_ic == p.oc && p.w_kw == int64_t(p.icg) * p.oc &&
                      p.w_kh == int64_t(p.kw) * p.icg * p.oc;
    }
    p.weight_vec = false;

    if (w_packed) {
        // The blob is what the panel would have gathered, so the geometry has to
        // agree to the byte: same groups, same channel count, same tiles.
        const ConvI8PackGeom pg = conv_i8_pack_geom(p.groups, p.icg, p.ocg, p.kh, p.kw);
        const char* const blob = wgt.ptr<const char>();
        if (!pg.ok || wgt.shape(0) != p.oc || wgt.numel() < pg.total ||
            (reinterpret_cast<uintptr_t>(blob) % 16 != 0)) {
            return Status::unsupported;
        }
        p.w_panel = blob;
        p.w_nblk = pg.nblk;
        p.r_b = reinterpret_cast<int32_t*>(const_cast<char*>(blob + pg.sums_off));
    }

    // The 16 B A-fill needs a dense channel axis and 16 channels of one tap per
    // vector (k/groups is tap-major there).
    const int kGroup = 16; // one 16 B vector of s8
    p.fill_vec = p.in_c == 1 && p.icg % kGroup == 0 && (p.in_w % 16 == 0) &&
                 (reinterpret_cast<uintptr_t>(p.input) % 16 == 0);

    if (conv_offset_span(p) >= (int64_t(1) << 30) || p.batch * p.groups > 65535 ||
        (p.ocg + 63) / 64 > 65535 || p.ocg <= 0) {
        return Status::unsupported;
    }

    const QuantParams& qa = act.quant_params();
    const QuantParams& qw = wgt.quant_params();
    p.zp_x = qa.zero_point_data != nullptr ? qa.zero_point_data[0] : qa.zero_point;
    p.sc_x = qa.scale_data != nullptr ? qa.scale_data[0] : qa.scale;
    p.zp_w_scalar = qw.zero_point;
    p.sc_w_scalar = qw.scale;

    const bool int_out = kind == I8OutKind::S32 || kind == I8OutKind::S8 || kind == I8OutKind::U8;
    p.relu = int_out && attrs.epilogue.type == EpilogueActivateType::Relu;
    if (int_out) {
        const QuantParams& qc = output.quant_params();
        p.out_scale = qc.scale_data != nullptr ? qc.scale_data[0] : qc.scale;
        p.out_zp = qc.zero_point_data != nullptr ? qc.zero_point_data[0] : qc.zero_point;
    }
    p.out_kind = static_cast<int>(kind);

    // ---- workspace: R_b and the staged per-channel arrays -----------------
    // A packed filter brings its own sums, so it needs no R_b plane.
    const int64_t rb_off = 0;
    int64_t cursor = w_packed ? 0 : int64_t(p.oc) * 4;
    int64_t off_zpw = -1, off_scw = -1;
    if (qw.zero_point_data != nullptr) {
        off_zpw = cursor;
        cursor += (int64_t(p.oc) * 4 + 15) & ~int64_t(15);
    }
    if (qw.scale_data != nullptr) {
        off_scw = cursor;
        cursor += (int64_t(p.oc) * 4 + 15) & ~int64_t(15);
    }

    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    const detail::GemmWorkspace ws =
        cursor > 0 ? detail::gemm_workspace_alloc(cursor, stream) : detail::GemmWorkspace{};
    if (cursor > 0 && ws.ptr == nullptr) {
        return Status::internal;
    }
    char* base = static_cast<char*>(ws.ptr);
    if (!w_packed) {
        p.r_b = reinterpret_cast<int32_t*>(base + rb_off);
    }
    const auto copy_in = [&](const void* src, int64_t off) {
        if (off < 0) {
            return true;
        }
        return cudaMemcpyAsync(base + off, src, size_t(int64_t(p.oc) * 4), cudaMemcpyHostToDevice,
                               stream) == cudaSuccess;
    };
    if (!copy_in(qw.zero_point_data, off_zpw) || !copy_in(qw.scale_data, off_scw)) {
        detail::gemm_workspace_free(ws, stream);
        cudaGetLastError();
        return Status::internal;
    }
    p.zp_w = off_zpw >= 0 ? reinterpret_cast<const int32_t*>(base + off_zpw) : nullptr;
    p.sc_w = off_scw >= 0 ? reinterpret_cast<const float*>(base + off_scw) : nullptr;

    // A blob carries the sums its pack pass wrote, so the reduction — and its
    // grid — is skipped rather than run and overwritten.
    if (!w_packed) {
        constexpr int kSumThreads = 128;
        const dim3 sgrid(unsigned((p.oc + kSumThreads - 1) / kSumThreads), 1, 1);
        conv_i8_col_sums_kernel<<<sgrid, kSumThreads, 0, stream>>>(p);
    }

    if (adt == DataType::u8) {
        launch_conv_i8<uint8_t>(p, stream);
    } else {
        launch_conv_i8<int8_t>(p, stream);
    }

    detail::gemm_workspace_free(ws, stream);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

// ============================================================
// the int8 route's prepacked filter, for Conv2D::prepackWeights
// ============================================================

static_assert(kConvI8PackBN == ConvShapeI8::BN && kConvI8PackBK == ConvShapeI8::BK &&
                  kConvI8PackLD == ConvShapeI8::LD,
              "the blob's panels are the kernel's shared-memory panels");

TensorDesc conv_i8_packed_weight_desc(const Conv2DAttributes& attrs, const TensorDesc& weight)
{
    TensorDesc d;
    if (weight.rank != 4 || weight.dtype != DataType::s8 ||
        weight.layout != TensorLayout::HWIO || attrs.algorithm == ConvAlgorithm::Winograd ||
        attrs.groups <= 0 || weight.dims[3] % attrs.groups != 0) {
        return d;
    }
    const int64_t oc = weight.dims[3];
    const ConvI8PackGeom g =
        conv_i8_pack_geom(attrs.groups, weight.dims[2], oc / attrs.groups, weight.dims[0],
                          weight.dims[1]);
    if (!g.ok) {
        return d;
    }
    d.rank = 2;
    d.layout = TensorLayout::PackedHWIOCuda;
    d.dtype = DataType::s8;
    d.dims[0] = oc;
    d.dims[1] = (g.total + oc - 1) / oc;
    d.row_stride_elems = d.dims[1];
    return d;
}

Status conv_i8_packed_weight_pack(const Conv2DAttributes& attrs, const TensorView& weight,
                                  TensorView& out, const ComputeContext& ctx)
{
    if (weight.rank() != 4 || weight.data_type() != DataType::s8 ||
        weight.layout() != TensorLayout::HWIO || attrs.groups <= 0 ||
        weight.shape(3) % attrs.groups != 0) {
        return Status::unsupported;
    }
    const int64_t oc = weight.shape(3);
    const ConvI8PackGeom g =
        conv_i8_pack_geom(attrs.groups, weight.shape(2), oc / attrs.groups, weight.shape(0),
                          weight.shape(1));
    if (!g.ok || out.rank() != 2 || out.data_type() != DataType::s8 ||
        out.layout() != TensorLayout::PackedHWIOCuda || out.shape(0) != oc ||
        out.numel() < g.total ||
        (reinterpret_cast<uintptr_t>(out.ptr<const void>()) % 16 != 0)) {
        return Status::unsupported;
    }

    ConvI8PackParams q;
    q.weight = weight.ptr<const void>();
    q.blob = out.ptr<void>();
    q.g = g;
    q.w_kh = int32_t(weight.stride_elems(0));
    q.w_kw = int32_t(weight.stride_elems(1));
    q.w_ic = int32_t(weight.stride_elems(2));
    q.w_n = int32_t(weight.stride_elems(3));

    const int64_t chunks = int64_t(g.groups) * g.ktiles * g.nblk * kConvI8PackBN *
                           (kConvI8PackBK / 16);
    const int64_t work = chunks > oc ? chunks : oc;
    const int threads = 256;
    const int blocks = int((work + threads - 1) / threads);
    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    conv_i8_pack_kernel<<<blocks, threads, 0, stream>>>(q);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda
