/// @file conv2d_implicit_gemm.cpp
/// @brief Conv2D as implicit GEMM over NHWC activations.
///
/// M = pixels, N = oc block, K tap-major. The gathered activation tile is A and
/// the packed HWIO weight is B, either prepacked once per call (shared arena
/// region) or per task.
///
/// One axis selects the family: the weight's dtype (fp or quantized, in
/// conv2d_gemm_run). Everything else — the plan prologue, the pooled arena, the
/// column split, the (activation, output) dtype ladders — is written once. The
/// decomposition is (batch, group) × oh-chunks × oc-chunks. Output tiles are
/// disjoint, so the route is bit-identical to serial execution.
///
/// What not to "fix" later:
///
///  * The first icn block BUILDS the tile — zero_mode, or the bias as the
///    accumulator's initial value (tile_mma_direct's per-column col_bias, whose
///    lanes are output channels). Seeding the tile first costs a write plus a
///    read-back of the whole tile for one addend. Only the block that builds it
///    may be seeded: a later one would re-seed and drop the partial sum. add_to
///    is the exception — it folds the *original* output, which no accumulator
///    seed can express.
///  * A pure Relu/Relu6 folds into the MMA store's min/max, on the last icn
///    block only (partial sums accumulate unclamped). Measured 11% of the conv2d
///    row at one thread, bit-identical checkpoints. add_to adds the residual
///    *after* the activation, so it keeps the separate epilogue pass.
///  * The prepacked filter's geometry comes from conv_packed_weight, and
///    the plan ADOPTS its icn_block/ocn_block from there instead of choosing
///    again: prepackWeights built the panels against that function, with no
///    activation to resolve a plan from. A plan that picked its own block would
///    address panels the blob does not hold.
///  * An fp activation is quantized into a compact plane once per call, never
///    inside the gather: the gather repeats once per oc block, and only the
///    materialized plane lets a 1x1 conv take the pointwise route.
///  * On AArch64 SME a raw f32/f16/s8 filter with a wide oc block (four
///    64-column ZA tiles) takes the MOPA kernels: B is packed once per call into
///    the ZA layout, per (group, task oc start, ic block) — the task starts need
///    not be 64-aligned — and each pixel tile's A is packed into [m/16][k][16].
///    Below that width the per-tile A pack and the half-empty 64-column blocks
///    lose to the NEON panels, so the route stays there. A caller's PackedHWIO
///    blob is the NEON layout and stays there too; its SME form
///    (PackedHWIOSme) is produced by prepackWeights for the same wide-OC fp
///    filters, so the packed path needs no per-call B pack either.
///
/// Plan heuristics, measured at one thread unless stated (this host, 6P+8E):
///
///  * ic-block ladder: kdim (= ic_block × karea) in ~144–200 is the cost knob;
///    past it the A tile the block multiplies costs more than the passes it
///    saves. Every ic block pays a full read-modify-write of its output tile and
///    restarts the gather. A 3x3 at the 8 rung (kdim 72) is 2.5–4.4% behind the
///    16 rung; the wider filters move much less.
///  * oc-chunks use the nearest whole number to nt/plane_tasks, not the ceiling:
///    a 9-block plane at 20 threads wants 2 chunks, ceil picks 3 and the barrier
///    waits on a 7-task second wave (152 vs 226 GFLOPS, 17x17). The filter axis
///    splits on fp and quantized alike, with the accumulator per pass.
///  * a rows-per-pass cap (ph_block, 128 KiB of accumulator): a 56x56 full-tile
///    set is 800 KB and every pass went to DRAM. Pixel tiles flatten across row
///    boundaries when rows are dense, so a narrow plane stops paying MR-row
///    granularity per row. The shared prepack cap is 8 MiB; past it each task
///    packs its own panels.
///  * The quantized route's R_b walk over the weight is deliberately serial: a
///    second pool dispatch costs more than the pass (1.18x at 2T → 1.9x at 20T
///    on a 3x3 whose zero points are not the packer's).

#include "conv2d_implicit_gemm.h"
#include "backend/cpu/matmul/matmul_helper.h"           // uniform pack/MMA + panel strides
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp"              // available_sme + SME tile geometry
#endif
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"    // epilogue / dequant epilogues
#include "backend/cpu/simd_kernel/simd_im2col.hpp"      // tiled im2col gathers
#include "backend/cpu/simd_kernel/simd_int8_reduce.hpp" // int8_row_sum / int8_col_sum_slab
#include "backend/cpu/simd_kernel/simd_quant.hpp"       // quant_kernel::quantization
#include "backend/cpu/simd_kernel/simd_stem_conv.hpp"   // the tiny filter's direct 3x3
#include "backend/cpu/common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace nnops::backend::cpu {

/// The activations the fp plan admits: f32 and f16 on every arch, bf16 only
/// where its kernels exist. The quantized route's (narrower) activation list
/// stays in the plan functions, since quantizing a bf16 activation is not wired.
constexpr bool is_fp_activation_dtype(DataType d) noexcept
{
    return d == DataType::f32 || d == DataType::f16
#ifdef NNOPS_ARCH_AARCH64
           || d == DataType::bf16
#endif
        ;
}

// =========================================================================
//  Im2col route — pixels on M, the packed HWIO weight on N
// =========================================================================

namespace impl {
namespace {

using nnops::simd::s_load;
using nnops::simd::s_store;

namespace k = nnops::kernel; // pure SIMD kernels: the stem's direct 3x3

// ---- block-size heuristics ------------------------------------------------

/// Input-channel block: kdim = ic_block × karea lands in ~144–256, and the
/// packed-B panel (nr_max × kdim) stays near L1/L2. A 1x1 filter has neither a
/// gather nor a karea multiplier, so K runs four times as wide.
inline int64_t get_ic_block(int64_t karea, int64_t ic_per_group) noexcept
{
    if (karea == 1) {
        return 256;
    }
    // A narrow channel count takes itself whole even when the ladder would split
    // it: with every channel of a pixel in the block, one filter row's taps are
    // adjacent, so the gather copies one run instead of `KW` runs of two. That
    // is the 3-channel stem's whole loss (IC=3, 7x7).
    if (ic_per_group * karea <= 256) {
        return std::max<int64_t>(ic_per_group, 1);
    }
    if (karea <= 9) {
        return 16;
    }
    if (karea <= 25) {
        return 8;
    }
    if (karea <= 49) {
        return 2;
    }
    return 1;
}

constexpr int64_t kOcnBlock = 128; ///< output-channel tile (the GEMM's N)
/// An oc_per_group at or below this stays one block: every extra oc block costs
/// a second pass over the pixels (A pack, accumulator fill, epilogue). Splitting
/// 144 into 80 + 64 cost the int8 expand layer a quarter of its time.
constexpr int64_t kOcnSingleCap = 192;
constexpr int64_t kMBlock = 24; ///< pixel tile (the GEMM's M): 4 × mr_max(6)
/// The widest pixel tile any route picks: the int8 SME route takes 16 * sme_mr
/// (256) rows. Stack buffers indexed by the tile are fixed at this width so the
/// file still compiles with the SME kernels switched off.
constexpr int64_t kMaxMBlock = 256;
/// Pixels a task needs before it is worth splitting the plane again: two M
/// blocks, i.e. one full tile plus a second to hide its prologue.
constexpr int64_t kTaskMinPixels = 2 * kMBlock;
/// Bytes one accumulator pass may cover (the plan's rows-per-pass knob).
constexpr int64_t kPassAccBytes = 128 * 1024;
constexpr int64_t kABSlack = 8; ///< elements past the A tile (vector over-read)

/// Cap on the shared packed-weight buffer: written once per call, read by every
/// task, so a small plane (few pixels per task) need not pay the whole pack per
/// task. Past 8 MiB it no longer fits an L3 alongside the working set.
constexpr int64_t kPrepackCap = int64_t{8} << 20;

/// Cap on the per-slot add_to residual / int32 accumulator tile set (elements).
/// Rows are split further until it fits: a residual path costs parallelism,
/// never memory.
constexpr int64_t kTaskTileCap = int64_t{64} << 10;

// ---- tile helpers ---------------------------------------------------------

/// Seed a tile with the per-column bias, or add it into what is there. add_to
/// only: a lone bias is the accumulator's initial value (tile_mma_direct's
/// col_bias), which costs no tile traffic.
template <class T>
void tile_seed(T* NNOPS_RESTRICT tile, int ld, int mc, int nc, const T* NNOPS_RESTRICT bias,
               bool add) noexcept
{
    for (int m = 0; m < mc; ++m) {
        T* row = tile + static_cast<int64_t>(m) * ld;
        if (add) {
            for (int i = 0; i < nc; ++i) {
                s_store(&row[i], s_load(&row[i]) + s_load(&bias[i]));
            }
        } else {
            std::memcpy(row, bias, static_cast<size_t>(nc) * sizeof(T));
        }
    }
}

/// Zero a strided tile [mc][nc] (row stride ld) — the accumulator seed when the
/// add_to route has no bias to start from.
template <class T> void tile_zero(T* NNOPS_RESTRICT tile, int ld, int mc, int nc) noexcept
{
    for (int m = 0; m < mc; ++m) {
        T* row = tile + static_cast<int64_t>(m) * ld;
        for (int i = 0; i < nc; ++i) {
            row[i] = T{};
        }
    }
}

/// Copy a strided tile (row stride ld) into a dense [M][N] scratch.
template <class T>
void tile_save(T* NNOPS_RESTRICT dst, const T* NNOPS_RESTRICT src, int ld, int M, int N) noexcept
{
    for (int m = 0; m < M; ++m) {
        const T* s = src + static_cast<int64_t>(m) * ld;
        T* d = dst + static_cast<int64_t>(m) * N;
        for (int i = 0; i < N; ++i) {
            d[i] = s[i];
        }
    }
}

// ---- the packed-B route ---------------------------------------------------
//
// HWIO hands the packer its own order: the output channel is contiguous, so one
// k row of a panel is one OC-length run — no transposing packer, no staging, and
// a 1x1 filter packs in one call. With karea > 1 the k slice is karea rectangles,
// so one panel's rows are restaged tap-major first; the MMA still makes one call
// over the whole kdim. Staging is per *panel* (nr_max × kdim, L1-sized).
template <class T>
void pack_block(T* dst, int64_t panel_slot, T* stage, const T* w, int64_t w_ic_stride,
                int64_t w_tap_stride, int64_t ocnc, int64_t icnc, int karea, int64_t nr_max,
                int kdim) noexcept
{
    (void)nr_max;
    if (karea == 1) {
        tile_pack_rhs(false, static_cast<int>(ocnc), static_cast<int>(icnc), dst,
                      static_cast<int>(panel_slot), w, static_cast<int>(w_ic_stride), 1.0f);
        return;
    }

    for (int64_t p0 = 0; p0 < ocnc; p0 += nr_max) {
        const int64_t np = std::min(nr_max, ocnc - p0);
        for (int t = 0; t < karea; ++t) {
            for (int64_t i = 0; i < icnc; ++i) {
                std::memcpy(stage + (static_cast<int64_t>(t) * icnc + i) * nr_max,
                            w + t * w_tap_stride + i * w_ic_stride + p0,
                            static_cast<size_t>(np) * sizeof(T));
            }
        }
        tile_pack_rhs(false, static_cast<int>(np), kdim, dst + (p0 / nr_max) * panel_slot,
                      static_cast<int>(panel_slot), stage, static_cast<int>(nr_max), 1.0f);
    }
}

// ---- the tiny filter ------------------------------------------------------
//
// The kernels live in simd_kernel/simd_stem_conv.hpp; this is the task grid over
// them, one task per output row: a row is ~27 multiply-adds per output element,
// wide enough that the pool's per-task cost is noise.

template <class T, int IC, int K>
void stem_conv_plane(T* NNOPS_RESTRICT out, const T* NNOPS_RESTRICT in, const T* NNOPS_RESTRICT w,
                     const T* NNOPS_RESTRICT bias, int64_t N, int64_t IH, int64_t IW, int64_t OW,
                     int64_t OC, int64_t in_rs, int64_t out_rs, int64_t ib_step, int64_t ob_step,
                     int64_t OH, int SH, int SW, int PH, int PW, int DH, int DW, bool clamp,
                     float lo, float hi, const ComputeContext& ctx)
{
    ctx.cpu.run(0, N * OH, [&](int64_t t) {
        const int64_t n = t / OH;
        k::stem_conv_row<T, IC, K>(out + n * ob_step, in + n * ib_step, w, bias, IH, IW, OW, OC,
                                   in_rs, out_rs, t - n * OH, SH, SW, PH, PW, DH, DW, clamp, lo,
                                   hi);
    });
}

// ---- the fused im2col + A pack (SME) ---------------------------------------

#if defined(NNOPS_ARCH_SME)
/// The im2col gather written straight into the ZA A layout. A ZA row is one
/// pixel's channel run (NHWC, so it is contiguous) and the K columns come out
/// of the tile vertically, so the gather and the transpose are one pass
/// instead of a dense write, a ZA re-read and a second write.
///
/// f32 holds one K per lane. f16 packs a channel's K pair into one word, which
/// needs an even channel count so a pair never straddles a tap — the caller
/// checks that and falls back to the dense gather plus pack_a_f16_sme.
template <class T>
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void tile_im2col_pack_sme(T* dst, const T* in, int64_t in_row_stride, int64_t ic_stride,
                                 int64_t p0, int OW, int mc, int IH, int IW, int KH, int KW,
                                 int SH, int SW, int DH, int DW, int PH, int PW, int icnc,
                                 int karea) noexcept
{
    constexpr bool f32 = std::is_same_v<T, float>;
    constexpr int kRows = 2 * aarch64::sme_mr; // 32 pixels: one A block
    constexpr int kChanStep = f32 ? 16 : 32;   // channels one ZA tile row spans
    const svfloat32_t vzero = svdup_n_f32(0.0f);
    int64_t base[kRows];
    bool live[kRows];

    for (int t = 0; t < karea; ++t) {
        const int kh = t / KW;
        const int kw = t - kh * KW;
        for (int i = 0; i < mc; ++i) {
            const int64_t p = p0 + i;
            const int64_t oh = p / OW;
            const int64_t ow = p - oh * OW;
            const int64_t ih = oh * SH - PH + static_cast<int64_t>(kh) * DH;
            const int64_t iw = ow * SW - PW + static_cast<int64_t>(kw) * DW;
            live[i] = ih >= 0 && ih < IH && iw >= 0 && iw < IW;
            base[i] = live[i] ? ih * in_row_stride + iw * ic_stride : 0;
        }
        for (int c0 = 0; c0 < icnc; c0 += kChanStep) {
            const int lanes = f32 ? std::min(16, icnc - c0) :
                                    std::min(16, (icnc - c0) / 2);
            const svbool_t pc = svwhilelt_b32(0, lanes);
            const int k0 = t * icnc + c0;
            for (int pstart = 0; pstart < mc; pstart += aarch64::sme_mr) {
                const int n = std::min(aarch64::sme_mr, mc - pstart);
                for (int i = 0; i < n; ++i) {
                    if (live[pstart + i]) {
                        if constexpr (f32) {
                            svld1_hor_za32(0, static_cast<uint32_t>(i), pc,
                                           in + base[pstart + i] + c0);
                        } else {
                            svld1_hor_za32(0, static_cast<uint32_t>(i), pc,
                                           reinterpret_cast<const uint32_t*>(in) +
                                               (base[pstart + i] + c0) / 2);
                        }
                    } else {
                        svwrite_hor_za32_f32_m(0, static_cast<uint32_t>(i), pc, vzero);
                    }
                }
                const svbool_t pr = svwhilelt_b32(0, n);
                if constexpr (f32) {
                    for (int j = 0; j < lanes; ++j) {
                        svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                       dst + static_cast<int64_t>(k0 + j) * kRows + pstart);
                    }
                } else {
                    auto* const dw = reinterpret_cast<uint32_t*>(dst);
                    for (int j = 0; j < lanes; ++j) {
                        svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                       dw + static_cast<int64_t>(k0 / 2 + j) * kRows +
                                           pstart);
                    }
                }
            }
        }
    }
}

/// The quantized twin: a ZA row is one pixel's channel run (four channels a
/// word), so a tap's 64-byte run per pixel is one `ld1_hor_za32` into that
/// pixel's row, and the K groups come out of the tile vertically into
/// [m/16][k/4][64]. The gather and the A pack are one pass, as in the float
/// form — the source must already be signed (@p xr applied upstream) and four
/// bytes wide per k group, and pixels outside the input read @p fill.
__arm_locally_streaming __arm_new("za") __attribute__((target("+sme2")))
inline void tile_im2col_pack_i8_sme(int8_t* dst, const int8_t* in, int64_t in_row_stride,
                                    int64_t ic_stride, int8_t fill, int64_t p0, int OW, int mc,
                                    int IH, int IW, int KH, int KW, int SH, int SW, int DH,
                                    int DW, int PH, int PW, int icnc, int karea) noexcept
{
    constexpr int kRows = aarch64::sme_mr;      // 16 pixels: one A panel
    constexpr int kChanMax = 64;                // bytes one ZA tile row spans
    constexpr int kMaxPix = 16 * aarch64::sme_mr; // one tile's pixels
    const int64_t groups = static_cast<int64_t>(karea) * icnc / 4;
    int8_t fill_row[kChanMax];
    // Per pixel: the patch's top-left corner. A tap is then two adds and two
    // range tests, not a divide, per pixel.
    int32_t ih0[kMaxPix];
    int32_t iw0[kMaxPix];
    std::memset(fill_row, static_cast<int>(static_cast<uint8_t>(fill)), sizeof(fill_row));

    {
        int64_t oh = p0 / OW;
        int64_t ow = p0 - oh * OW;
        for (int i = 0; i < mc; ++i, ++ow) {
            if (ow == OW) {
                ow = 0;
                ++oh;
            }
            ih0[i] = static_cast<int32_t>(oh * SH - PH);
            iw0[i] = static_cast<int32_t>(ow * SW - PW);
        }
    }

    // A tile whose every pixel's whole patch lies inside the input is the common
    // case, and there a tap is one add per pixel off a base computed once —
    // no range tests, no multiplies. The bulk of this function's cost was that
    // addressing, not the ZA moves themselves.
    int64_t base0[kMaxPix];
    bool interior = true;
    {
        const int64_t hspan = static_cast<int64_t>(KH - 1) * DH;
        const int64_t wspan = static_cast<int64_t>(KW - 1) * DW;
        for (int i = 0; i < mc; ++i) {
            interior = interior && ih0[i] >= 0 && ih0[i] + hspan < IH && iw0[i] >= 0 &&
                       iw0[i] + wspan < IW;
            base0[i] = static_cast<int64_t>(ih0[i]) * in_row_stride +
                       static_cast<int64_t>(iw0[i]) * ic_stride;
        }
    }

    for (int t = 0; t < karea; ++t) {
        const int kh = t / KW;
        const int kw = t - kh * KW;
        const int trow = kh * DH;
        const int tcol = kw * DW;
        int64_t base[kMaxPix];
        bool live[kMaxPix];
        if (interior) {
            const int64_t shift = static_cast<int64_t>(trow) * in_row_stride +
                                  static_cast<int64_t>(tcol) * ic_stride;
            for (int i = 0; i < mc; ++i) {
                base[i] = base0[i] + shift;
            }
        } else {
            for (int i = 0; i < mc; ++i) {
                const int ih = ih0[i] + trow;
                const int iw = iw0[i] + tcol;
                live[i] = ih >= 0 && ih < IH && iw >= 0 && iw < IW;
                base[i] = live[i] ? static_cast<int64_t>(ih) * in_row_stride +
                                        static_cast<int64_t>(iw) * ic_stride :
                                    0;
            }
        }
        for (int c0 = 0; c0 < icnc; c0 += kChanMax) {
            const int words = std::min(kChanMax, icnc - c0) / 4;
            const svbool_t pc = svwhilelt_b32(0, words);
            for (int pstart = 0; pstart < mc; pstart += kRows) {
                const int n = std::min(kRows, mc - pstart);
                if (interior && n == kRows) {
                    // Constant row indices: the ZA slice folds into the
                    // instruction instead of a register, and the whole panel
                    // pipelines. Worth 10-25% of this function.
#pragma unroll
                    for (int i = 0; i < kRows; ++i) {
                        svld1_hor_za32(
                            0, static_cast<uint32_t>(i), pc,
                            reinterpret_cast<const uint32_t*>(in + base[pstart + i] + c0));
                    }
                } else if (interior) {
                    for (int i = 0; i < n; ++i) {
                        svld1_hor_za32(
                            0, static_cast<uint32_t>(i), pc,
                            reinterpret_cast<const uint32_t*>(in + base[pstart + i] + c0));
                    }
                } else {
                    for (int i = 0; i < n; ++i) {
                        const int8_t* const src = live[pstart + i] ?
                                                      in + base[pstart + i] + c0 :
                                                      fill_row;
                        svld1_hor_za32(0, static_cast<uint32_t>(i), pc,
                                       reinterpret_cast<const uint32_t*>(src));
                    }
                }
                const svbool_t pr = svwhilelt_b32(0, n);
                auto* const dw = reinterpret_cast<int32_t*>(dst);
                const int64_t blk = static_cast<int64_t>(pstart / kRows) * groups;
                const int64_t bjat = blk + static_cast<int64_t>(t) * icnc / 4 + c0 / 4;
                // A vertical ZA store costs about five cycles; reading four
                // columns at once and storing them as two 128-byte memory
                // writes is twice as fast over the whole pack. Four columns are
                // 256 contiguous bytes in the A panel, so the pair stores land
                // back to back.
                int j = 0;
                if (n == kRows) {
                    for (; j + 4 <= words; j += 4) {
                        const svint32x4_t g = svread_ver_za32_s32_vg4(0, static_cast<uint32_t>(j));
                        svst1_s32_x4(svptrue_c32(), dw + (bjat + j) * aarch64::sme_mr, g);
                    }
                }
                for (; j < words; ++j) {
                    svst1_ver_za32(0, static_cast<uint32_t>(j), pr,
                                   dw + (bjat + j) * aarch64::sme_mr);
                }
            }
        }
    }
}
#endif

// ---- the per-call context -------------------------------------------------
//
// Everything the task grid reads, resolved once per call. The task and its
// tile helpers take this by reference; the alternative — a lambda capturing
// forty locals — is what made this kernel one function of six hundred lines.

/// The once-per-call SME B prepack: one ZA-layout copy per (group, task oc
/// start, ic block), so a task reads its filter in place. `start_prefix` maps
/// an oc chunk to its first start; blocks inside a region are `slot_max`
/// elements apart.
template <class T> struct SmeFpShared {
    std::vector<int64_t> start_prefix;
    PoolArray<T> b;
    PoolArray<T> stage;
    int64_t total_starts = 0;
    int64_t region_elems = 0;
    int64_t slot_max = 0;
    int64_t num_ic_blocks = 0;
};

template <class T> struct FpConvCtx {
    static constexpr int64_t nr_max = nr_max_flt<T>();

    const Conv2DAttributes* attrs = nullptr;
    const ComputeContext* ctx = nullptr;
    const T* in_ptr = nullptr;
    const T* w_ptr = nullptr;
    const T* b_ptr = nullptr;
    T* out_ptr = nullptr;
    bool has_bias = false;

    int64_t N = 0, IC = 0, IH = 0, IW = 0, OC = 0, OH = 0, OW = 0;
    int64_t in_row_stride = 0, out_row_stride = 0;
    int64_t G = 0, ic_per_group = 0, oc_per_group = 0;
    int KH = 0, KW = 0, SH = 0, SW = 0, DH = 0, DW = 0, PH = 0, PW = 0, karea = 0;

    int64_t icn_block = 0, ocn_block = 0, oh_chunks = 0, ph_block = 0, oc_chunks = 0;
    int m_block = 0;
    int64_t ib_step = 0, ob_step = 0;
    int64_t w_ic_stride = 0, w_tap_stride = 0;
    int64_t esz = 0;
    bool prepacked = false, sme_blob = false, use_sme = false, pointwise = false;

    float clamp_lo = 0.0f, clamp_hi = 0.0f;
    bool epi_folded = false, epi_pass = false;
    bool keep_orig = false, seed_bias = false, bias_seed = false, need_epi = false;

    ConvPackedWeight pg;
    int64_t num_ic_blocks = 0, kdim_max = 0;
    Conv2DGemmScratch sc;
    char* base = nullptr;
    T* shared_ptr = nullptr;
    bool shared = false;
    const T* panels = nullptr;
    int64_t total = 0;
    bool per_thread = false;
    PoolArray<T> bias_pad;
    SmeFpShared<T> sme;
};

/// The finish decision: what folds into the MMA store and what needs a pass.
struct FpFinish {
    float clamp_lo = 0.0f;
    float clamp_hi = 0.0f;
    bool epi_folded = false;
    bool epi_pass = false;
    bool keep_orig = false;
    bool seed_bias = false;
    bool bias_seed = false;
    bool need_epi = false;
};

inline FpFinish fp_finish(const Conv2DAttributes& attrs, bool has_bias) noexcept
{
    constexpr float kInf = std::numeric_limits<float>::infinity();
    const bool pre_act = attrs.add_to && attrs.residual_pre_activation;

    FpFinish f;
    f.clamp_lo = -kInf;
    f.clamp_hi = kInf;
    if (!attrs.add_to || pre_act) {
        switch (attrs.epilogue.type) {
        case EpilogueActivateType::Relu:
            f.clamp_lo = 0.0f;
            f.epi_folded = true;
            break;
        case EpilogueActivateType::Relu6:
            f.clamp_lo = 0.0f;
            f.clamp_hi = 6.0f;
            f.epi_folded = true;
            break;
        default:
            break;
        }
    }
    const bool epi_active = attrs.epilogue.type != EpilogueActivateType::None;
    f.epi_pass = epi_active && !f.epi_folded;
    // add_to folds the pre-existing output into the accumulator seed when the
    // epilogue is the identity, and the pre-activation order always does; the
    // post-activation order with an activation preserves the original tile and
    // adds it after the epilogue instead.
    f.keep_orig = attrs.add_to && f.epi_pass && !pre_act;
    // The bias is per column, so it always seeds the tile; add_to's seed also
    // carries the residual. A seeded tile must never be zeroed by the MMA.
    f.seed_bias = attrs.add_to && has_bias;
    f.bias_seed = has_bias && !attrs.add_to;
    f.need_epi = f.epi_pass;
    return f;
}

/// The tiny filter: at most four input channels under a square 3x3/5x5/7x7
/// takes the direct plane kernel — no gather, no panel, no plan. prepackWeights
/// declines this shape too, so the filter arrives raw. Returns true when the
/// call was handled.
template <class T> bool try_stem(const FpConvCtx<T>& c)
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        if (c.G == 1 && !c.prepacked && c.ic_per_group <= 4 && c.oc_per_group % 16 == 0 &&
            c.KH == c.KW && c.KH >= 3 && c.KH <= 7 && (c.KH & 1) && !c.attrs->add_to &&
            !c.epi_pass) {
            constexpr float kInf = std::numeric_limits<float>::infinity();
            const bool clamp = c.clamp_lo != -kInf || c.clamp_hi != kInf;
            const float lo = clamp ? c.clamp_lo : -kInf;
            const float hi = clamp ? c.clamp_hi : kInf;
            const auto run = [&](auto kk) {
                constexpr int K = decltype(kk)::value;
                switch (c.ic_per_group) {
                case 1:
                    stem_conv_plane<T, 1, K>(c.out_ptr, c.in_ptr, c.w_ptr, c.b_ptr, c.N, c.IH,
                                             c.IW, c.OW, c.OC, c.in_row_stride, c.out_row_stride,
                                             c.ib_step, c.ob_step, c.OH, c.SH, c.SW, c.PH, c.PW,
                                             c.DH, c.DW, clamp, lo, hi, *c.ctx);
                    return;
                case 2:
                    stem_conv_plane<T, 2, K>(c.out_ptr, c.in_ptr, c.w_ptr, c.b_ptr, c.N, c.IH,
                                             c.IW, c.OW, c.OC, c.in_row_stride, c.out_row_stride,
                                             c.ib_step, c.ob_step, c.OH, c.SH, c.SW, c.PH, c.PW,
                                             c.DH, c.DW, clamp, lo, hi, *c.ctx);
                    return;
                case 3:
                    stem_conv_plane<T, 3, K>(c.out_ptr, c.in_ptr, c.w_ptr, c.b_ptr, c.N, c.IH,
                                             c.IW, c.OW, c.OC, c.in_row_stride, c.out_row_stride,
                                             c.ib_step, c.ob_step, c.OH, c.SH, c.SW, c.PH, c.PW,
                                             c.DH, c.DW, clamp, lo, hi, *c.ctx);
                    return;
                default:
                    stem_conv_plane<T, 4, K>(c.out_ptr, c.in_ptr, c.w_ptr, c.b_ptr, c.N, c.IH,
                                             c.IW, c.OW, c.OC, c.in_row_stride, c.out_row_stride,
                                             c.ib_step, c.ob_step, c.OH, c.SH, c.SW, c.PH, c.PW,
                                             c.DH, c.DW, clamp, lo, hi, *c.ctx);
                    return;
                }
            };
            switch (c.KH) {
            case 3:
                run(std::integral_constant<int, 3>{});
                return true;
            case 5:
                run(std::integral_constant<int, 5>{});
                return true;
            default:
                run(std::integral_constant<int, 7>{});
                return true;
            }
        }
    }
    return false;
}

/// The shared SME filter prepack (once per call). Blocks inside a region are
/// `slot_max` apart, and the start list is the tasks' own oc tiling —
/// `ocn_block` need not be a 64-column multiple, so a per-chunk layout could
/// not be addressed.
template <class T> void pack_sme_filter_shared(FpConvCtx<T>& c, SmeFpShared<T>& s)
{
#if defined(NNOPS_ARCH_SME)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        if (!c.use_sme || c.sme_blob) {
            return;
        }
        constexpr DataType be_dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
        const int64_t nr = aarch64::sme_fp_nr;
        s.num_ic_blocks = (c.ic_per_group + c.icn_block - 1) / c.icn_block;
        s.start_prefix.assign(static_cast<size_t>(c.oc_chunks) + 1, 0);
        for (int64_t chunk = 0; chunk < c.oc_chunks; ++chunk) {
            const int64_t lo = chunk * c.pg.panels_oc / c.oc_chunks * FpConvCtx<T>::nr_max;
            const int64_t hi = std::min((chunk + 1) * c.pg.panels_oc / c.oc_chunks *
                                            FpConvCtx<T>::nr_max,
                                        c.oc_per_group);
            const int64_t starts = (hi - lo + c.ocn_block - 1) / c.ocn_block;
            s.start_prefix[static_cast<size_t>(chunk) + 1] =
                s.start_prefix[static_cast<size_t>(chunk)] + starts;
        }
        s.total_starts = s.start_prefix[static_cast<size_t>(c.oc_chunks)];
        s.slot_max = sme_b_panel_stride(be_dt, c.kdim_max);
        const int64_t max_blocks = (c.ocn_block + nr - 1) / nr;
        s.region_elems = max_blocks * s.slot_max;
        const int64_t slices = c.G * s.total_starts * s.num_ic_blocks;
        s.b.reset(static_cast<size_t>(slices * s.region_elems));
        if (c.karea > 1) {
            s.stage.reset(static_cast<size_t>(c.kdim_max * nr));
        }
        for (int64_t g = 0; g < c.G; ++g) {
            const T* const w_group = c.w_ptr + g * c.oc_per_group;
            for (int64_t chunk = 0; chunk < c.oc_chunks; ++chunk) {
                const int64_t oc_lo =
                    chunk * c.pg.panels_oc / c.oc_chunks * FpConvCtx<T>::nr_max;
                const int64_t oc_hi =
                    std::min((chunk + 1) * c.pg.panels_oc / c.oc_chunks *
                                 FpConvCtx<T>::nr_max,
                             c.oc_per_group);
                int64_t si = 0;
                for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += c.ocn_block, ++si) {
                    const int64_t ocnc = std::min(c.ocn_block, oc_hi - ocn);
                    for (int64_t ib = 0; ib < s.num_ic_blocks; ++ib) {
                        const int64_t icn0 = ib * c.icn_block;
                        const int64_t icnc = std::min(c.icn_block, c.ic_per_group - icn0);
                        const int64_t kdim = icnc * c.karea;
                        T* const dst =
                            s.b.data() +
                            ((g * s.total_starts + s.start_prefix[static_cast<size_t>(chunk)] +
                              si) *
                                 s.num_ic_blocks +
                             ib) *
                                s.region_elems;
                        for (int64_t p0 = 0; p0 < ocnc; p0 += nr) {
                            const int64_t np = std::min(nr, ocnc - p0);
                            const T* const bsrc = w_group + ocn + p0 + icn0 * c.w_ic_stride;
                            T* const out = dst + (p0 / nr) * s.slot_max;
                            if (c.karea == 1) {
                                tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out, bsrc,
                                                  static_cast<int>(c.w_ic_stride));
                            } else {
                                for (int t = 0; t < c.karea; ++t) {
                                    for (int64_t i = 0; i < icnc; ++i) {
                                        std::memcpy(
                                            s.stage.data() +
                                                (static_cast<int64_t>(t) * icnc + i) * nr,
                                            bsrc + t * c.w_tap_stride + i * c.w_ic_stride,
                                            static_cast<size_t>(np) * sizeof(T));
                                    }
                                }
                                tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out,
                                                  s.stage.data(), static_cast<int>(nr));
                            }
                        }
                    }
                }
            }
        }
    }
#else
    (void)c;
    (void)s;
#endif
}

/// Resolve the context. Returns false when the stem kernel handled the call.
template <class T>
bool init_fp_ctx(FpConvCtx<T>& c, const Conv2DAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx,
                 const Conv2DGemmPlan& plan, void* workspace)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    c.attrs = &attrs;
    c.ctx = &ctx;
    c.has_bias = inputs.size() > 2;

    // Channel-last [N, H, W, C]: a "row" is one spatial line and a pixel's
    // channel vector is contiguous.
    c.N = input.shape(0);
    c.IC = input.shape(3);
    c.IH = input.shape(1);
    c.IW = input.shape(2);
    c.in_row_stride = input.row_stride_elems();

    // HWIO: the output channel is the last axis; a prepacked blob leads with it.
    c.sme_blob = weight.layout() == TensorLayout::PackedHWIOSme;
    c.prepacked = is_packed_conv_weight(weight.layout());
    c.OC = c.prepacked ? weight.shape(0) : weight.shape(3);

    c.OH = output.shape(1);
    c.OW = output.shape(2);
    c.out_row_stride = output.row_stride_elems();

    c.G = attrs.groups;
    c.ic_per_group = c.IC / c.G;
    c.oc_per_group = c.OC / c.G;

#if defined(NNOPS_ARCH_SME)
    // Only wide-OC blocks fill four 64-column ZA tiles; below that the per-tile
    // A pack and the half-empty B blocks lose to the NEON panels. A caller's
    // prepacked HWIO blob is the NEON layout and stays there too.
    constexpr bool sme_dtype = std::is_same_v<T, float> || std::is_same_v<T, half>;
    c.use_sme = sme_dtype && (c.sme_blob || (!c.prepacked && aarch64::available_sme() &&
                                             c.oc_per_group >= aarch64::sme_fp_nr));
#else
    c.use_sme = false;
#endif

    c.KH = static_cast<int>(attrs.kernel_size[0]);
    c.KW = static_cast<int>(attrs.kernel_size[1]);
    c.SH = static_cast<int>(attrs.stride[0]);
    c.SW = static_cast<int>(attrs.stride[1]);
    c.DH = static_cast<int>(attrs.dilation[0]);
    c.DW = static_cast<int>(attrs.dilation[1]);
    c.PH = static_cast<int>(attrs.padding[0]);
    c.PW = static_cast<int>(attrs.padding[1]);
    c.karea = c.KH * c.KW;

    c.pointwise = plan.pointwise;
    c.icn_block = plan.icn_block;
    c.ocn_block = plan.ocn_block;
    c.m_block = static_cast<int>(plan.m_block);
    c.oh_chunks = plan.oh_chunks;
    c.ph_block = plan.ph_block;
    c.oc_chunks = plan.oc_chunks;

    c.ib_step = c.IH * c.in_row_stride;
    c.ob_step = c.OH * c.out_row_stride;

    // HWIO (TensorFlow's filter): [KH][KW][IC/G][OC]. The output channel is the
    // contiguous axis, so one (tap, input channel) is an OC-length run and the B
    // panel is a plain strided copy — no transpose. Dims, not pitch.
    c.w_ic_stride = c.prepacked ? 0 : weight.shape(3);
    c.w_tap_stride = c.prepacked ? 0 : weight.shape(2) * c.w_ic_stride;

    c.out_ptr = output.ptr<T>();
    c.in_ptr = input.ptr<T>();
    c.w_ptr = weight.ptr<T>();
    c.b_ptr = c.has_bias ? inputs[2].ptr<T>() : nullptr;
    c.esz = static_cast<int64_t>(sizeof(T));

    const FpFinish fin = fp_finish(attrs, c.has_bias);
    c.clamp_lo = fin.clamp_lo;
    c.clamp_hi = fin.clamp_hi;
    c.epi_folded = fin.epi_folded;
    c.epi_pass = fin.epi_pass;
    c.keep_orig = fin.keep_orig;
    c.seed_bias = fin.seed_bias;
    c.bias_seed = fin.bias_seed;
    c.need_epi = fin.need_epi;

    if (try_stem<T>(c)) {
        return false;
    }

    // ---- block geometry ----------------------------------------------------
    // Same function prepackWeights built the blob with; it must never disagree.
    c.pg = c.sme_blob ? conv_packed_weight_sme(attrs, c.ic_per_group, c.oc_per_group,
                                               weight.data_type()) :
                        conv_packed_weight(attrs, c.ic_per_group, c.oc_per_group,
                                           weight.data_type());
    if (!c.sme_blob) {
        NNOPS_ASSERT(c.pg.icn_block == c.icn_block && c.pg.ocn_block == c.ocn_block &&
                     c.pg.nr == FpConvCtx<T>::nr_max);
    }
    c.num_ic_blocks = c.pg.nfull_ic + (c.pg.tail_ic ? 1 : 0);
    c.kdim_max = c.pg.kdim_max;

    // ---- scratch -----------------------------------------------------------
    c.sc = gemm_scratch_layout(plan, c.esz);
    c.base = static_cast<char*>(workspace);
    c.shared_ptr = reinterpret_cast<T*>(c.base + c.sc.panel_off);
    c.shared = plan.shared_elems > 0;
    // The panels every task reads in place: the caller's blob, or this call's
    // prepack in the arena (null when the weight did not fit — each task then
    // packs its own into its slot).
    c.panels =
        (!c.use_sme && (c.prepacked || c.shared)) ? (c.prepacked ? c.w_ptr : c.shared_ptr) :
                                                    nullptr;

    // ---- the shared weight prepack (once per call) -------------------------
    if (c.shared && !c.use_sme) {
        const int64_t cols = (plan.shared_elems + c.OC - 1) / c.OC;
        const int64_t shape[] = {c.OC, cols};
        TensorView blob(std::span<const int64_t>(shape, 2), weight.data_type(), c.shared_ptr,
                        cols * c.esz, TensorLayout::PackedHWIO);
        conv_packed_weight_pack(attrs, weight, blob, ctx);
    }

    c.total = (c.N * c.G) * (c.oh_chunks * c.oc_chunks);
    const int nslots = ctx.cpu.thread_slot_count();
    c.per_thread = nslots > 0 && nslots < c.total;

    // The micro-kernel loads a bias panel at a time, so a group's bias must be
    // readable `nr_max` past its last channel. One padded copy covers every
    // group: lanes past a group are only read, and the store drops them.
    if (c.bias_seed) {
        c.bias_pad.assign(static_cast<size_t>(c.OC + FpConvCtx<T>::nr_max), T{});
        std::memcpy(c.bias_pad.data(), c.b_ptr,
                    static_cast<size_t>(c.OC) * sizeof(T));
    }

    pack_sme_filter_shared(c, c.sme);
    return true;
}

/// The B panel a task reads: in place from a blob or a 1x1 filter, or packed
/// into its slot. The SME form carries a block stride instead of an ld.
template <class T> struct FpBPanel {
    const T* ptr = nullptr;
    int ld = 0;
    int64_t sme_stride = 0;
};

template <class T>
FpBPanel<T> fp_pick_b(const FpConvCtx<T>& c, int64_t g, int64_t icn, int64_t icnc,
                      int kdim, int64_t oc_chunk, int64_t ocn, int64_t ocnc, T* stage,
                      T* pack_buf)
{
    FpBPanel<T> b;
#if defined(NNOPS_ARCH_SME)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        if (c.use_sme) {
            if (c.sme_blob) {
                // The prepacked blob's block for (group, ic block, 64-column oc
                // block); consecutive oc blocks are one region apart.
                const int64_t slice =
                    (g * c.pg.num_ic_blocks + icn / c.icn_block) * c.pg.num_oc_blocks +
                    ocn / c.pg.nr;
                b.ptr = c.w_ptr + slice * c.pg.sme_region;
                b.sme_stride = c.pg.sme_region;
            } else {
                // The shared prepack's slice for this task's (group, oc start,
                // ic block); its first block is the task's own `ocn`.
                const int64_t oc_lo_chunk =
                    oc_chunk * c.pg.panels_oc / c.oc_chunks * FpConvCtx<T>::nr_max;
                const int64_t starts_before =
                    c.sme.start_prefix[static_cast<size_t>(oc_chunk)];
                const int64_t si_local = (ocn - oc_lo_chunk) / c.ocn_block;
                const int64_t slice =
                    (g * c.sme.total_starts + starts_before + si_local) *
                        c.sme.num_ic_blocks +
                    icn / c.icn_block;
                b.ptr = c.sme.b.data() + slice * c.sme.region_elems;
                b.sme_stride = c.sme.slot_max;
            }
            return b;
        }
    }
#endif
    const T* const w_g = c.w_ptr + g * c.oc_per_group;
    if (c.panels) {
        // The blob's own walk: this oc block's k block, in place.
        b.ptr = c.panels + c.pg.offset(g, icn / c.icn_block, ocn);
    } else if (c.karea == 1) {
        // A 1x1 filter is [ic][oc] with oc contiguous — a panel's k rows
        // already, `w_ic_stride` apart — so read it in place.
        b.ptr = w_g + ocn + icn * c.w_ic_stride;
        b.ld = static_cast<int>(c.w_ic_stride);
    } else {
        pack_block<T>(pack_buf,
                      gemm_panel_slot(kdim, FpConvCtx<T>::nr_max, c.esz, false,
                                      std::is_same_v<T, bfloat16>),
                      stage, w_g + ocn + icn * c.w_ic_stride, c.w_ic_stride, c.w_tap_stride,
                      ocnc, icnc, c.karea, FpConvCtx<T>::nr_max, kdim);
        b.ptr = pack_buf;
    }
    return b;
}

/// The A tile a pixel tile multiplies: the input run itself (pointwise), the
/// fused ZA gather on SME, or the dense im2col scratch.
template <class T> struct FpATile {
    const T* ptr = nullptr;
    int lda = 0;
    bool packed = false;
};

template <class T>
FpATile<T> fp_pick_a(const FpConvCtx<T>& c, const T* in_g, int64_t icn, int64_t icnc,
                     int64_t p, int64_t oh, int64_t ow0, int mc, bool flat, T* a_buf,
                     T* sme_a)
{
    FpATile<T> a;
    const int kdim = static_cast<int>(icnc * c.karea);
    if (c.pointwise) {
        a.ptr = in_g + oh * c.in_row_stride + ow0 * c.IC + icn;
        a.lda = static_cast<int>(c.IC);
        return a;
    }
    bool fused = false;
#if defined(NNOPS_ARCH_SME)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        // f16 pairs two K values in a word: an odd channel run would put a pair
        // across two taps.
        fused = c.use_sme && (c.IC % 2 == 0) && (c.in_row_stride % 2 == 0) &&
                (!std::is_same_v<T, half> || (icnc % 2) == 0);
    }
#endif
    if (fused) {
#if defined(NNOPS_ARCH_SME)
        // The ZA route's gather writes the packed A layout directly, so it needs
        // no separate transpose pass.
        tile_im2col_pack_sme<T>(sme_a, in_g + icn, c.in_row_stride, c.IC, p,
                                static_cast<int>(c.OW), mc, static_cast<int>(c.IH),
                                static_cast<int>(c.IW), c.KH, c.KW, c.SH, c.SW, c.DH, c.DW,
                                c.PH, c.PW, static_cast<int>(icnc), c.karea);
        a.ptr = sme_a;
        a.lda = kdim;
        a.packed = true;
#endif
    } else {
        tiled_im2col<T>(a_buf, kdim, in_g + icn, c.in_row_stride, c.IC, T{}, p,
                        static_cast<int>(c.OW), flat, mc, static_cast<int>(c.IH),
                        static_cast<int>(c.IW), c.KH, c.KW, c.SH, c.SW, c.DH, c.DW, c.PH,
                        c.PW, static_cast<int>(icnc), c.karea);
        a.ptr = a_buf;
        a.lda = kdim;
    }
    return a;
}

/// One pixel tile: seed (add_to), multiply, and finish. The first icn block
/// builds the tile; later blocks accumulate.
template <class T>
void fp_mma_tile(const FpConvCtx<T>& c, int64_t g, int64_t icn, int64_t icnc, bool is_last_icn,
                 int mc, int64_t ocn, int64_t ocnc, T* tile, T* orig_tile,
                 const FpATile<T>& a, const FpBPanel<T>& b, T* sme_a, T* stage)
{
    const int kdim = static_cast<int>(icnc * c.karea);
    const int64_t ldc = c.OC;
    const T* const bias_g = c.b_ptr ? c.b_ptr + g * c.oc_per_group : nullptr;
    const T* const bias_col =
        c.bias_seed ? c.bias_pad.data() + g * c.oc_per_group : nullptr;

    if (icn == 0 && c.attrs->add_to) {
        if (c.keep_orig) {
            tile_save<T>(orig_tile, tile, static_cast<int>(ldc), mc, static_cast<int>(ocnc));
        }
        if (c.seed_bias) {
            // The residual stays in the accumulator only on the identity
            // epilogue route; keep_orig saves it and the tile is the seed.
            tile_seed<T>(tile, static_cast<int>(ldc), mc, static_cast<int>(ocnc),
                         bias_g + ocn, /*add=*/!c.keep_orig);
        } else if (c.keep_orig) {
            // No bias: the tile holds the residual, which the epilogue adds
            // back itself.
            tile_zero<T>(tile, static_cast<int>(ldc), mc, static_cast<int>(ocnc));
        }
    }

    constexpr float kInf = std::numeric_limits<float>::infinity();
    const bool clamp_here = is_last_icn && c.epi_folded;
    const float c_lo = clamp_here ? c.clamp_lo : -kInf;
    const float c_hi = clamp_here ? c.clamp_hi : kInf;
    const bool zero_mode = icn == 0 && !c.attrs->add_to && !c.bias_seed;

    bool mma_sme = false;
#if defined(NNOPS_ARCH_SME)
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        if (c.use_sme) {
            if (!a.packed) {
                tile_pack_lhs_sme(false, mc, kdim, sme_a, a.ptr, a.lda);
            }
            tile_mma_pack_sme(mc, static_cast<int>(ocnc), kdim, tile, static_cast<int>(ldc),
                              sme_a, b.ptr, b.sme_stride, c_lo, c_hi, zero_mode,
                              (icn == 0 && c.bias_seed) ? bias_col + ocn : nullptr);
            mma_sme = true;
        }
    }
#endif
    if (!mma_sme) {
        tile_mma_direct(mc, static_cast<int>(ocnc), kdim, tile, static_cast<int>(ldc), a.ptr,
                        a.lda, b.ptr, b.ld != 0 ? b.ld : -1, stage, c_lo, c_hi, zero_mode,
                        nullptr,
                        (icn == 0 && c.bias_seed) ? bias_col + ocn : nullptr);
    }

    if (is_last_icn && c.need_epi) {
        epilogue_inplace<T>(mc, static_cast<int>(ocnc), tile, static_cast<int>(ldc), nullptr,
                            c.attrs->epilogue, c.keep_orig ? orig_tile : nullptr, c.keep_orig);
    }
}

/// One task of the (batch, group) × oh-chunk × oc-chunk grid.
template <class T> void run_fp_task(const FpConvCtx<T>& c, int64_t t)
{
    const int64_t chunks_per_ng = c.oh_chunks * c.oc_chunks;
    const int64_t ng = t / chunks_per_ng;
    const int64_t rem = t % chunks_per_ng;
    // The oc-chunk varies fastest: neighbours in task order then cover
    // different filter rows, so their weight streams differ.
    const int64_t chunk = rem / c.oc_chunks;
    const int64_t oc_chunk = rem % c.oc_chunks;
    const int64_t n = ng / c.G;
    const int64_t g = ng % c.G;

    const int64_t slot =
        c.per_thread ? static_cast<int64_t>(c.ctx->cpu.current_thread_id()) : t;
    char* const slot_ptr = c.base + c.sc.head + slot * c.sc.slot_stride;
    T* const a_buf = reinterpret_cast<T*>(slot_ptr + c.sc.off_a);
    T* const orig_buf = reinterpret_cast<T*>(slot_ptr + c.sc.off_orig);
    T* const stage = reinterpret_cast<T*>(slot_ptr + c.sc.off_stage);
    T* const pack_buf = reinterpret_cast<T*>(slot_ptr + c.sc.off_pack);

    // The SME route's per-task A slab; B is the shared prepack.
    PoolArray<T> sme_a;
#if defined(NNOPS_ARCH_SME)
    if (c.use_sme) {
        constexpr int64_t kSmeARows = 2 * aarch64::sme_mr;
        const int64_t a_blocks = (c.m_block + kSmeARows - 1) / kSmeARows;
        sme_a.reset(static_cast<size_t>(a_blocks * sme_a_panel_stride(
                                             std::is_same_v<T, float> ? DataType::f32 :
                                                                        DataType::f16,
                                             c.kdim_max)));
    }
#endif

    T* const out_g = c.out_ptr + n * c.ob_step + g * c.oc_per_group;
    const T* const in_g = c.in_ptr + n * c.ib_step + g * c.ic_per_group;

    // This oh-chunk's contiguous row range, and its oc range. The oc split is
    // over *panels*, not channels: the blob holds panel p as channels
    // [p·nr, (p+1)·nr) of its k column, so a task starts where a panel does.
    const int64_t r0 = chunk * c.OH / c.oh_chunks;
    const int64_t r1 = (chunk + 1) * c.OH / c.oh_chunks;
    // An SME blob keeps the whole group on the filter axis (its plan sets
    // oc_chunks 1): starts and the 64-column blocks then agree.
    const int64_t oc_lo =
        c.sme_blob ? 0 : oc_chunk * c.pg.panels_oc / c.oc_chunks * FpConvCtx<T>::nr_max;
    const int64_t oc_hi =
        c.sme_blob ? c.oc_per_group :
                     std::min((oc_chunk + 1) * c.pg.panels_oc / c.oc_chunks *
                                  FpConvCtx<T>::nr_max,
                              c.oc_per_group);

    // Dense rows make a pixel tile one contiguous run, so it may cross a row
    // boundary — which keeps a narrow plane's tiles at the full M. Pointwise
    // reads A the same way, so the input must be dense.
    const bool flat =
        c.out_row_stride == c.OW * c.OC && (!c.pointwise || c.in_row_stride == c.IW * c.IC);

    // A task may start mid-block (oc_chunks splits on panels), but the blob
    // holds one oc block at a time: end the tile where the block does, so its
    // panels keep the stride the kernel derives from kdim.
    const auto block_end = [&](int64_t ocn) {
        return c.panels ? std::min(oc_hi, (ocn / c.ocn_block + 1) * c.ocn_block) : oc_hi;
    };
    for (int64_t ocn = oc_lo; ocn < oc_hi;
         ocn = c.panels ? block_end(ocn) : ocn + c.ocn_block) {
        const int64_t oc_end = block_end(ocn);
        const int64_t ocnc = std::min(c.ocn_block, oc_end - ocn);
        T* const orig = c.keep_orig ? orig_buf : nullptr;

        for (int64_t pr = r0; pr < r1; pr += c.ph_block) {
            const int64_t pr1 = std::min(r1, pr + c.ph_block);
            const int64_t p_first = pr * c.OW;
            const int64_t p_last = pr1 * c.OW;
            // A short tail is re-split, keeping the step a multiple of MR.
            constexpr int64_t mr_step = mr_max_flt<T>();
            const int64_t p_span = std::max<int64_t>(p_last - p_first, 1);
            const int64_t p_tail = p_span % c.m_block;
            const int64_t p_step =
                (flat && p_tail != 0 && p_tail < mr_step && c.m_block >= 2 * mr_step) ?
                    (c.m_block / mr_step - 1) * mr_step :
                    c.m_block;

            for (int64_t icn = 0; icn < c.ic_per_group; icn += c.icn_block) {
                const int64_t icnc = std::min(c.icn_block, c.ic_per_group - icn);
                const bool is_last_icn = (icn + c.icn_block) >= c.ic_per_group;
                const int kdim = static_cast<int>(icnc * c.karea);

                const FpBPanel<T> b =
                    fp_pick_b(c, g, icn, icnc, kdim, oc_chunk, ocn, ocnc, stage, pack_buf);

                for (int64_t p = p_first; p < p_last;) {
                    const int64_t oh = p / c.OW;
                    const int64_t ow0 = p - oh * c.OW;
                    const int64_t left =
                        flat ? std::min(p_step, p_last - p) : (c.OW - ow0);
                    const int mc = static_cast<int>(std::min<int64_t>(c.m_block, left));

                    const FpATile<T> a = fp_pick_a(c, in_g, icn, icnc, p, oh, ow0, mc, flat,
                                                  a_buf, sme_a.data());
                    T* const tile = out_g + oh * c.out_row_stride + ow0 * c.OC + ocn;
                    T* const orig_tile =
                        orig ? orig + (p - p_first) * c.ocn_block : nullptr;

                    fp_mma_tile(c, g, icn, icnc, is_last_icn, mc, ocn, ocnc, tile, orig_tile,
                                a, b, sme_a.data(), stage);
                    p += flat ? p_step : mc;
                }
            }
        }
    }
}

// ---- main kernel ----------------------------------------------------------

template <class T>
void conv2d_impl(const Conv2DAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx,
                 const Conv2DGemmPlan& plan, void* workspace)
{
    FpConvCtx<T> c;
    if (!init_fp_ctx(c, attrs, output, inputs, ctx, plan, workspace)) {
        return;
    }
    ctx.cpu.run(0, c.total, [&](int64_t t) { run_fp_task(c, t); });
}

// ---- the quantized route --------------------------------------------------
//
// Integer operands: A the packed activation tile (bytes, A shift applied), B
// the packed s8 HWIO weight, int32 accumulator (one read-modify-write pass per
// ic block over the task's oc slice); the epilogue compensates and dequantizes
// or requantizes. An fp activation is materialized into a compact s8 plane once
// per call.

namespace k = nnops::kernel; // pure SIMD kernels: the quantizer, int8 reductions

constexpr int64_t align4(int64_t v) noexcept { return (v + 3) & ~int64_t{3}; }

/// True when any zero point the tensor carries is non-zero.
inline bool any_zp_nonzero(const QuantParams& q, int64_t n) noexcept
{
    if (q.zero_point_data == nullptr) {
        return q.zero_point != 0;
    }
    for (int64_t i = 0; i < n; ++i) {
        if (q.zero_point_data[i] != 0) {
            return true;
        }
    }
    return false;
}

/// Build the compact s8 activation plane: dense [N][H][W][IC], walking the
/// source's row pitch. @p zero already carries the A shift (a u8 target enters
/// zp − 128), so the quantizer's clamp lands signed bytes and nothing XORs
/// afterwards; a pre-quantized u8 copy gets that XOR, an s8 one is read in place.
template <class TAct>
void materialize_plane(const TensorView& in, int8_t* dst, float scale, int32_t zero,
                       const ComputeContext& ctx)
{
    constexpr bool kPreQuant = std::is_same_v<TAct, int8_t> || std::is_same_v<TAct, uint8_t>;

    const int64_t IC = in.shape(3);
    const int64_t IH = in.shape(1);
    const int64_t IW = in.shape(2);
    const int64_t N = in.shape(0);
    const int64_t rs = in.row_stride_elems();
    const TAct* src = in.ptr<TAct>();

    const float s = scale;
    const float z = static_cast<float>(zero);

    const int64_t row_len = IW * IC;
    const int64_t rows = N * IH;
    const int64_t total = rows * row_len;
    if (total == 0) {
        return;
    }

    auto plane_range = [&](int64_t beg, int64_t end) {
        for (int64_t e = beg; e < end;) {
            const int64_t row = e / row_len;
            const int64_t off = e - row * row_len;
            const int64_t len = std::min(row_len - off, end - e);
            int8_t* d = dst + e;
            const TAct* p = src + row * rs + off;
            if constexpr (kPreQuant) {
                int64_t j = 0;
#if defined(__ARM_NEON)
                const uint8x16_t m = vdupq_n_u8(0x80);
                for (; j + 16 <= len; j += 16) {
                    vst1q_u8(reinterpret_cast<uint8_t*>(d + j),
                             veorq_u8(vld1q_u8(reinterpret_cast<const uint8_t*>(p + j)), m));
                }
#endif
                for (; j < len; ++j) {
                    d[j] = static_cast<int8_t>(static_cast<uint8_t>(p[j]) ^ 0x80u);
                }
            } else {
                k::quant_kernel::quantization<int8_t>(static_cast<int>(1), static_cast<int>(len), d,
                                                      1, p, 1, &s, &z);
            }
            e += len;
        }
    };

    // A dispatch costs more than the pass.
    constexpr int64_t kInlineMax = int64_t{1} << 19;

    const int64_t nt = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    if (nt == 1 || total <= kInlineMax) {
        plane_range(0, total);
        return;
    }
    const int64_t slab = (total + nt - 1) / nt;
    const int64_t nslab = (total + slab - 1) / slab;
    ctx.cpu.run(0, nslab,
                [&](int64_t i) { plane_range(i * slab, std::min(i * slab + slab, total)); });
}

// ---- the packed-B route ---------------------------------------------------

/// pack_block's integer twin: same panel, same tap-major restaging.
///
/// @p a_biased must be set whenever the kernel folds an A-side bias out of its
/// accumulators through the panel's column sums — on x86 always, since that is
/// where the +128 the VNNI kernel adds to every A group is cancelled.
inline void pack_block_i8(int8_t* dst, int64_t panel_slot, int8_t* stage, const int8_t* w,
                          int64_t w_ic_stride, int64_t w_tap_stride, int64_t ocnc, int64_t icnc,
                          int karea, int64_t nr_max, int kdim, bool a_biased) noexcept
{
    if (karea == 1) {
        tile_pack_rhs_i8(false, static_cast<int>(ocnc), static_cast<int>(icnc), dst,
                         static_cast<int>(panel_slot), w, static_cast<int>(w_ic_stride), a_biased,
                         false);
        return;
    }

    for (int64_t p0 = 0; p0 < ocnc; p0 += nr_max) {
        const int64_t np = std::min(nr_max, ocnc - p0);
        for (int t = 0; t < karea; ++t) {
            for (int64_t i = 0; i < icnc; ++i) {
                std::memcpy(stage + (static_cast<int64_t>(t) * icnc + i) * nr_max,
                            w + t * w_tap_stride + i * w_ic_stride + p0, static_cast<size_t>(np));
            }
        }
        tile_pack_rhs_i8(false, static_cast<int>(np), kdim, dst + (p0 / nr_max) * panel_slot,
                         static_cast<int>(panel_slot), stage, static_cast<int>(nr_max), a_biased,
                         false);
    }
}

// ---- the per-call context -------------------------------------------------

/// The once-per-call SME int8 B prepack; see SmeFpShared.
struct SmeI8Shared {
    std::vector<int64_t> start_prefix;
    PoolArray<int8_t> b;
    PoolArray<int8_t> stage;
    int64_t total_starts = 0;
    int64_t region_bytes = 0;
    int64_t slot_max = 0;
    int64_t num_ic_blocks = 0;
};

template <class TAct, class TOut> struct I8ConvCtx {
    static constexpr bool kFpAct =
        !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    static constexpr bool kU8 = std::is_same_v<TAct, uint8_t>;
    static constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;
    static constexpr int64_t nr_max = NR_MAX_I8;

    const Conv2DAttributes* attrs = nullptr;
    const ComputeContext* ctx = nullptr;
    const QuantParams* qw = nullptr;
    const QuantParams* qc = nullptr;
    const TOut* b_ptr = nullptr;
    const int32_t* b_i32 = nullptr;
    TOut* out_ptr = nullptr;
    const int8_t* w_ptr = nullptr;
    const uint8_t* in_base = nullptr;
    int64_t in_row_stride = 0;

    int64_t N = 0, IC = 0, IH = 0, IW = 0, OC = 0, OH = 0, OW = 0;
    int64_t out_row_stride = 0;
    int64_t G = 0, ic_per_group = 0, oc_per_group = 0;
    int KH = 0, KW = 0, SH = 0, SW = 0, DH = 0, DW = 0, PH = 0, PW = 0, karea = 0;
    int64_t k_full = 0;

    int64_t icn_block = 0, ocn_block = 0, oh_chunks = 0, ph_block = 0, oc_chunks = 0;
    int m_block = 0;
    int64_t ib_step = 0, ob_step = 0;
    int64_t w_ic_stride = 0, w_tap_stride = 0;
    bool prepacked = false, sme_blob = false, use_sme = false, pointwise = false;

    float s_x = 0.0f;
    int32_t zp_x = 0;
    bool u8a = false;
    int32_t d = 0;
    bool col_term = false, comp = false;
    bool plane = false;
    uint8_t xr = 0;

    ConvPackedWeight pg;
    int64_t kdim_max = 0;
    bool a_biased = false;
    Conv2DGemmScratch sc;
    char* base = nullptr;
    int8_t* shared_ptr = nullptr;
    bool shared = false;
    const int8_t* panels = nullptr;
    PoolArray<int32_t> r_b_all;

    int64_t a_pack_stride = 0; ///< packed-A panels at the front of the A region
    int64_t acc_size = 0; ///< per-slot int32 accumulator elements
    int64_t total = 0;
    bool per_thread = false;

    bool epi_fused = false, epi_i8 = false;
    float out_scale = 0.0f;
    int32_t out_zp = 0;
    float epi_lo = 0.0f, epi_hi = 0.0f;

    SmeI8Shared sme;
};

/// Materialize the activation as s8 bytes (or point at it directly).
template <class TAct, class TOut>
void prepare_i8_input(I8ConvCtx<TAct, TOut>& c, const TensorView& input,
                      const ComputeContext& ctx)
{
    if (c.plane) {
        // s8 bytes, compact: the quantizer (or the pre-quantized copy) carries
        // the shift. The zero point moves into the *byte* domain too — a u8
        // source's `q − 128` is `q + d` — or the quantizer would clamp a
        // u8-range zero against the s8 clamp.
        int8_t* p = reinterpret_cast<int8_t*>(c.base + c.sc.quant_off);
        materialize_plane<TAct>(input, p, c.s_x, c.zp_x + c.d, ctx);
        c.in_base = reinterpret_cast<const uint8_t*>(p);
        c.in_row_stride = c.IW * c.IC;
        c.xr = 0;
    } else {
        c.in_base = reinterpret_cast<const uint8_t*>(input.ptr<TAct>());
        c.in_row_stride = input.row_stride_elems();
    }
    c.ib_step = c.IH * c.in_row_stride;
}

/// R_b[oc] = Σ_k weight[oc,k] over the whole K. A packed blob carries it, so
/// the walk below is only for a filter too large for the arena, and it is
/// serial because a second pool dispatch costs more than the pass.
///
/// In HWIO one output channel's K values are a *strided* column, so the sum
/// accumulates over the k rows — elementwise across OC, the contiguous axis.
template <class TAct, class TOut> void compute_i8_r_b(I8ConvCtx<TAct, TOut>& c)
{
    if (!c.col_term) {
        return;
    }
    c.r_b_all.assign(static_cast<size_t>(c.OC), 0);
    if (c.panels) {
        std::memcpy(c.r_b_all.data(), c.panels + c.pg.sums_off,
                    static_cast<size_t>(c.OC) * sizeof(int32_t));
        return;
    }
    for (int t = 0; t < c.karea; ++t) {
        for (int64_t i = 0; i < c.ic_per_group; ++i) {
            const int8_t* const row = c.w_ptr + t * c.w_tap_stride + i * c.w_ic_stride;
            for (int64_t oc = 0; oc < c.OC; ++oc) {
                c.r_b_all[static_cast<size_t>(oc)] += row[oc];
            }
        }
    }
}

/// The shared SME int8 filter prepack. Per (group, task oc start, ic block),
/// the same layout the fp route uses; `ocn_block` need not be a 64-column
/// multiple, so the region is cut at the tasks' own oc steps.
template <class TAct, class TOut> void pack_i8_sme_filter_shared(I8ConvCtx<TAct, TOut>& c)
{
#if defined(NNOPS_ARCH_SME)
    if (!c.use_sme) {
        return;
    }
    if (c.sme_blob) {
        // The blob holds one region per (group, ic block, 64-column block),
        // whole-K: nothing to pack here.
        c.sme.num_ic_blocks = (c.ic_per_group + c.icn_block - 1) / c.icn_block;
        c.sme.slot_max = c.pg.sme_region;
        return;
    }
    const int64_t nr = aarch64::sme_nr_max;
    c.sme.num_ic_blocks = (c.ic_per_group + c.icn_block - 1) / c.icn_block;
    c.sme.start_prefix.assign(static_cast<size_t>(c.oc_chunks) + 1, 0);
    for (int64_t chunk = 0; chunk < c.oc_chunks; ++chunk) {
        const int64_t lo = chunk * c.pg.panels_oc / c.oc_chunks * I8ConvCtx<TAct, TOut>::nr_max;
        const int64_t hi = std::min((chunk + 1) * c.pg.panels_oc / c.oc_chunks *
                                        I8ConvCtx<TAct, TOut>::nr_max,
                                    c.oc_per_group);
        const int64_t starts = (hi - lo + c.ocn_block - 1) / c.ocn_block;
        c.sme.start_prefix[static_cast<size_t>(chunk) + 1] =
            c.sme.start_prefix[static_cast<size_t>(chunk)] + starts;
    }
    c.sme.total_starts = c.sme.start_prefix[static_cast<size_t>(c.oc_chunks)];
    c.sme.slot_max = sme_b_panel_stride(DataType::s8, c.kdim_max);
    const int64_t max_blocks = (c.ocn_block + nr - 1) / nr;
    c.sme.region_bytes = max_blocks * c.sme.slot_max;
    const int64_t slices = c.G * c.sme.total_starts * c.sme.num_ic_blocks;
    c.sme.b.reset(static_cast<size_t>(slices * c.sme.region_bytes));
    if (c.karea > 1) {
        c.sme.stage.reset(static_cast<size_t>(c.kdim_max * nr));
    }
    for (int64_t g = 0; g < c.G; ++g) {
        const int8_t* const w_group = c.w_ptr + g * c.oc_per_group;
        for (int64_t chunk = 0; chunk < c.oc_chunks; ++chunk) {
            const int64_t oc_lo =
                chunk * c.pg.panels_oc / c.oc_chunks * I8ConvCtx<TAct, TOut>::nr_max;
            const int64_t oc_hi =
                std::min((chunk + 1) * c.pg.panels_oc / c.oc_chunks *
                             I8ConvCtx<TAct, TOut>::nr_max,
                         c.oc_per_group);
            int64_t si = 0;
            for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += c.ocn_block, ++si) {
                const int64_t ocnc = std::min(c.ocn_block, oc_hi - ocn);
                for (int64_t ib = 0; ib < c.sme.num_ic_blocks; ++ib) {
                    const int64_t icn0 = ib * c.icn_block;
                    const int64_t icnc = std::min(c.icn_block, c.ic_per_group - icn0);
                    const int64_t kdim = icnc * c.karea;
                    int8_t* const dst =
                        c.sme.b.data() +
                        ((g * c.sme.total_starts +
                          c.sme.start_prefix[static_cast<size_t>(chunk)] + si) *
                             c.sme.num_ic_blocks +
                         ib) *
                            c.sme.region_bytes;
                    for (int64_t p0 = 0; p0 < ocnc; p0 += nr) {
                        const int64_t np = std::min(nr, ocnc - p0);
                        const int8_t* const bsrc = w_group + ocn + p0 + icn0 * c.w_ic_stride;
                        int8_t* const out = dst + (p0 / nr) * c.sme.slot_max;
                        if (c.karea == 1) {
                            tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out, bsrc,
                                                 static_cast<int>(c.w_ic_stride));
                        } else {
                            for (int t = 0; t < c.karea; ++t) {
                                for (int64_t i = 0; i < icnc; ++i) {
                                    std::memcpy(
                                        c.sme.stage.data() +
                                            (static_cast<int64_t>(t) * icnc + i) * nr,
                                        bsrc + t * c.w_tap_stride + i * c.w_ic_stride,
                                        static_cast<size_t>(np));
                                }
                            }
                            tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out,
                                                 c.sme.stage.data(), static_cast<int>(nr));
                        }
                    }
                }
            }
        }
    }
#else
    (void)c;
#endif
}

template <class TAct, class TOut>
void init_i8_ctx(I8ConvCtx<TAct, TOut>& c, const Conv2DAttributes& attrs, TensorView& output,
                 std::span<const TensorView> inputs, const ComputeContext& ctx,
                 const Conv2DGemmPlan& plan, void* workspace)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;
    c.attrs = &attrs;
    c.ctx = &ctx;

    c.N = input.shape(0);
    c.IC = input.shape(3);
    c.IH = input.shape(1);
    c.IW = input.shape(2);

    // HWIO: the output channel is the last axis; a prepacked blob leads with it.
    c.prepacked = is_packed_conv_weight(weight.layout());
    c.sme_blob = weight.layout() == TensorLayout::PackedHWIOSme;
    c.OC = c.prepacked ? weight.shape(0) : weight.shape(3);

    c.OH = output.shape(1);
    c.OW = output.shape(2);
    c.out_row_stride = output.row_stride_elems();

    c.G = attrs.groups;
    c.ic_per_group = c.IC / c.G;
    c.oc_per_group = c.OC / c.G;

    // The SMOPA tiles are 16 rows x 64 columns: one pass is four 16-column
    // groups, so the filter has to fill them. A caller's NEON panel blob has no
    // reader here; the ZA blob does, at the same 64-column width.
#if defined(NNOPS_ARCH_SME)
    c.use_sme = c.sme_blob || (!c.prepacked && aarch64::available_sme() &&
                               c.oc_per_group >= aarch64::sme_nr_max);
#else
    c.use_sme = false;
#endif

    c.KH = static_cast<int>(attrs.kernel_size[0]);
    c.KW = static_cast<int>(attrs.kernel_size[1]);
    c.SH = static_cast<int>(attrs.stride[0]);
    c.SW = static_cast<int>(attrs.stride[1]);
    c.DH = static_cast<int>(attrs.dilation[0]);
    c.DW = static_cast<int>(attrs.dilation[1]);
    c.PH = static_cast<int>(attrs.padding[0]);
    c.PW = static_cast<int>(attrs.padding[1]);
    c.karea = c.KH * c.KW;
    c.k_full = c.ic_per_group * c.karea;

    c.pointwise = plan.pointwise;
    c.icn_block = plan.icn_block;
    c.ocn_block = plan.ocn_block;
    c.m_block = static_cast<int>(plan.m_block);
    c.oh_chunks = plan.oh_chunks;
    c.ph_block = plan.ph_block;
    c.oc_chunks = plan.oc_chunks;
    c.ob_step = c.OH * c.out_row_stride;

    // HWIO: dims, not pitch — one (tap, input channel) is one OC-length run.
    c.w_ic_stride = c.prepacked ? 0 : weight.shape(3);
    c.w_tap_stride = c.prepacked ? 0 : weight.shape(2) * c.w_ic_stride;

    // Same geometry prepackWeights used (see the float kernel).
    c.pg = c.sme_blob ? conv_packed_weight_sme(attrs, c.ic_per_group, c.oc_per_group,
                                               weight.data_type()) :
                        conv_packed_weight(attrs, c.ic_per_group, c.oc_per_group,
                                           weight.data_type());
    NNOPS_ASSERT(c.pg.icn_block == c.icn_block && c.pg.ocn_block == c.ocn_block &&
                 c.pg.nr == NR_MAX_I8);
    c.kdim_max = c.pg.kdim_max;
    c.a_biased = arch::mma_i8_folds_a_bias;

    // ---- quantization parameters -----------------------------------------
    const QuantParams& qa = input.quant_params();
    c.qw = &weight.quant_params();
    c.qc = &output.quant_params();
    c.s_x = quant_scale_at(qa, 0);
    c.zp_x = quant_zp_at(qa, 0);
    NNOPS_ASSERT(c.s_x > 0.0f);

    // `d`: the shift the A bytes carry into the GEMM. Both arches fold out only
    // the +128 *they* add, so the epilogue must undo `d`. Which applies comes
    // from the *quantization target* — the activation's own storage type when
    // it arrives quantized, else `act_quant_dtype`.
    c.u8a = I8ConvCtx<TAct, TOut>::kFpAct ? (attrs.act_quant_dtype == DataType::u8) :
                                            I8ConvCtx<TAct, TOut>::kU8;
    c.d = c.u8a ? -128 : 0;
    // The column term exists when the A-side shift and zero point do not
    // cancel; the per-pixel term when some weight zero point is non-zero.
    c.col_term = (c.d + c.zp_x) != 0;
    c.comp = any_zp_nonzero(*c.qw, c.OC);

    // ---- the activation as s8 bytes ---------------------------------------
    c.sc = gemm_scratch_layout(plan, 1);
    c.base = static_cast<char*>(workspace);
    c.plane = plan.quant_size > 0;
    c.xr = c.u8a ? uint8_t{0x80} : uint8_t{0};

    c.w_ptr = weight.ptr<int8_t>();
    c.b_ptr = (has_bias && !I8ConvCtx<TAct, TOut>::kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    c.b_i32 = (has_bias && I8ConvCtx<TAct, TOut>::kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    c.out_ptr = output.ptr<TOut>();

    // ---- block geometry ----------------------------------------------------
    c.shared_ptr = reinterpret_cast<int8_t*>(c.base + c.sc.panel_off);
    c.shared = plan.shared_elems > 0;
    // The panels every task reads in place: the caller's blob, or this call's
    // own copy (null when the weight did not fit — each task packs its slot).
    c.panels = (!c.use_sme && (c.prepacked || c.shared)) ?
                   (c.prepacked ? c.w_ptr : c.shared_ptr) :
                   nullptr;

    // ---- the shared weight prepack (once per call) -------------------------
    // Through the entry prepackWeights uses, so the addressing is one code path.
    // It runs before the sums are read back below. The SME route packs its own
    // B layout per task, so this NEON copy is skipped there.
    if (c.shared && !c.use_sme) {
        const int64_t cols = (plan.shared_elems + c.OC - 1) / c.OC;
        const int64_t shape[] = {c.OC, cols};
        TensorView blob(std::span<const int64_t>(shape, 2), weight.data_type(), c.shared_ptr,
                        cols, TensorLayout::PackedHWIO);
        conv_packed_weight_pack(attrs, weight, blob, ctx);
    }

    compute_i8_r_b(c);
    pack_i8_sme_filter_shared(c);
    prepare_i8_input(c, input, ctx);

    c.a_pack_stride = plan.a_pack_stride;
    c.acc_size = plan.acc_size;
    c.total = (c.N * c.G) * (c.oh_chunks * c.oc_chunks);
    const int nslots = ctx.cpu.thread_slot_count();
    c.per_thread = nslots > 0 && nslots < c.total;

    // ---- the epilogue route -------------------------------------------------
    // The dequantizing store folds the whole epilogue into the kernel's
    // write-back. It needs the plain dequant shape — no per-row or per-column
    // compensation, a clamp-only activation, no residual — and one k block per
    // tile, so every tile starts from a zeroed ZA.
#if defined(NNOPS_ARCH_SME)
    c.epi_fused = c.use_sme && !I8ConvCtx<TAct, TOut>::kIntOut && !c.comp && !c.col_term &&
                  plan.orig_size == 0 && c.icn_block >= c.ic_per_group &&
                  attrs.epilogue.quant == EpilogueQuantType::None &&
                  (attrs.epilogue.type == EpilogueActivateType::None ||
                   attrs.epilogue.type == EpilogueActivateType::Relu ||
                   attrs.epilogue.type == EpilogueActivateType::Relu6);
    // The 8-bit requantizing store: same preconditions, but the destination is
    // quantized, so the epilogue's requantize rides the write-back too.
    c.epi_i8 = c.use_sme && I8ConvCtx<TAct, TOut>::kIntOut && !c.comp && !c.col_term &&
               plan.orig_size == 0 && c.icn_block >= c.ic_per_group &&
               (attrs.epilogue.quant != EpilogueQuantType::Requantize ||
                attrs.epilogue.quant_param_count == 0) &&
               (attrs.epilogue.type == EpilogueActivateType::None ||
                attrs.epilogue.type == EpilogueActivateType::Relu);
#else
    c.epi_fused = false;
    c.epi_i8 = false;
#endif
    // The requantize scale: s_a / s_out, and the destination's zero point —
    // what int8_epilogue_row_quant forms per row when nothing overrides it.
    c.out_scale = (attrs.epilogue.quant == EpilogueQuantType::Requantize &&
                   attrs.epilogue.quant_param_count > 0) ?
                      epilogue_quant_scale(attrs.epilogue, 0) :
                      quant_scale_at(*c.qc, 0);
    c.out_zp = (attrs.epilogue.quant == EpilogueQuantType::Requantize &&
                attrs.epilogue.quant_param_count > 0) ?
                   epilogue_quant_zero(attrs.epilogue, 0) :
                   quant_zp_at(*c.qc, 0);

    // Relu and Relu6 compose into the clip bounds: clamp∘clamp is a clamp.
    c.epi_lo = attrs.epilogue.min_clip;
    c.epi_hi = attrs.epilogue.max_clip;
    if (attrs.epilogue.type == EpilogueActivateType::Relu) {
        c.epi_lo = std::max(c.epi_lo, 0.0f);
    } else if (attrs.epilogue.type == EpilogueActivateType::Relu6) {
        c.epi_lo = std::max(c.epi_lo, 0.0f);
        c.epi_hi = std::min(c.epi_hi, 6.0f);
    }
}

/// The B panel a task reads: in place from a blob, or packed into its slot.
template <class TAct, class TOut>
const int8_t* pick_i8_b(const I8ConvCtx<TAct, TOut>& c, int64_t g, int64_t icn, int64_t ocn,
                        int64_t ocnc, int64_t icnc, int kdim, int64_t oc_chunk,
                        int8_t* stage, int8_t* pack_buf)
{
#if defined(NNOPS_ARCH_SME)
    if (c.use_sme) {
        if (c.sme_blob) {
            // The caller's blob: one whole-K region per (group, ic block,
            // 64-column block), read in place.
            const int64_t slice = (g * c.sme.num_ic_blocks + icn / c.icn_block) *
                                      c.pg.num_oc_blocks +
                                  ocn / aarch64::sme_nr_max;
            return c.w_ptr + slice * c.pg.sme_region;
        }
        const int64_t oc_lo_chunk =
            oc_chunk * c.pg.panels_oc / c.oc_chunks * I8ConvCtx<TAct, TOut>::nr_max;
        const int64_t starts_before = c.sme.start_prefix[static_cast<size_t>(oc_chunk)];
        const int64_t si_local = (ocn - oc_lo_chunk) / c.ocn_block;
        const int64_t slice = (g * c.sme.total_starts + starts_before + si_local) *
                                  c.sme.num_ic_blocks +
                              icn / c.icn_block;
        return c.sme.b.data() + slice * c.sme.region_bytes;
    }
#endif
    const int8_t* const w_g = c.w_ptr + g * c.oc_per_group;
    if (c.panels) {
        return c.panels + c.pg.offset(g, icn / c.icn_block, ocn);
    }
    pack_block_i8(pack_buf,
                  gemm_panel_slot(kdim, I8ConvCtx<TAct, TOut>::nr_max, 1, true), stage,
                  w_g + ocn + icn * c.w_ic_stride, c.w_ic_stride, c.w_tap_stride, ocnc, icnc,
                  c.karea, I8ConvCtx<TAct, TOut>::nr_max, kdim, c.a_biased);
    return pack_buf;
}

/// The A tile a pixel tile multiplies: the input run itself (pointwise), the
/// fused ZA gather on SME, or the dense int8 im2col scratch.
struct I8ATile {
    const int8_t* ptr = nullptr;
    int lda = 0;
    bool packed = false;
    bool direct = false;
};

template <class TAct, class TOut>
I8ATile pick_i8_a(const I8ConvCtx<TAct, TOut>& c, const uint8_t* in_g, const uint8_t* in_row,
                  int64_t icn, int64_t icnc, int64_t p, int64_t ow0, int mc, bool flat,
                  int kdim, int kdim4, int8_t* a_buf, int8_t* sme_a)
{
    I8ATile a;
    // Pointwise reads the pixels straight out of the row, which needs a whole
    // number of 4-byte k groups; else the gather zero-fills.
    a.direct = c.pointwise && (kdim == kdim4);
    if (a.direct) {
        // The pixel is its own patch: A is the input run.
        a.ptr = reinterpret_cast<const int8_t*>(in_row + ow0 * c.IC + icn);
        a.lda = static_cast<int>(c.IC);
        return a;
    }

    bool fused_pack = false;
#if defined(NNOPS_ARCH_SME)
    // The SME route gathers straight into the packed A panel (one ZA row per
    // pixel), which needs signed bytes in whole k groups — a u8 source comes in
    // shifted through the materialized plane.
    fused_pack = c.use_sme && c.xr == 0 && (icnc % 4) == 0 && (c.IC % 4) == 0 &&
                 (c.in_row_stride % 4) == 0 &&
                 (reinterpret_cast<uintptr_t>(in_g + icn) % 4) == 0;
#endif
    if (fused_pack) {
#if defined(NNOPS_ARCH_SME)
        tile_im2col_pack_i8_sme(
            sme_a, reinterpret_cast<const int8_t*>(in_g + icn), c.in_row_stride, c.IC,
            static_cast<int8_t>(static_cast<uint8_t>((c.plane ? c.zp_x + c.d : c.zp_x)) ^
                                c.xr),
            p, static_cast<int>(c.OW), mc, static_cast<int>(c.IH), static_cast<int>(c.IW),
            c.KH, c.KW, c.SH, c.SW, c.DH, c.DW, c.PH, c.PW, static_cast<int>(icnc), c.karea);
        a.ptr = sme_a;
        a.lda = kdim4;
        a.packed = true;
#endif
    } else {
        tiled_im2col_i8(a_buf, kdim4, in_g + icn, c.in_row_stride, c.IC, c.xr,
                        static_cast<uint8_t>(c.plane ? c.zp_x + c.d : c.zp_x), p,
                        static_cast<int>(c.OW), flat, mc, static_cast<int>(c.IH),
                        static_cast<int>(c.IW), c.KH, c.KW, c.SH, c.SW, c.DH, c.DW, c.PH,
                        c.PW, static_cast<int>(icnc), c.karea, kdim, kdim4);
        a.ptr = a_buf;
        a.lda = kdim4;
    }
    return a;
}

/// Multiply one pixel tile. Returns true when a fused store already wrote the
/// destination (no int32 tile was materialized).
template <class TAct, class TOut>
bool i8_mma(const I8ConvCtx<TAct, TOut>& c, int64_t g, int64_t icn, int mc, int64_t ocn,
            int64_t ocnc, int kdim4, int32_t* acc_tile, int8_t* a_pack, const I8ATile& a,
            const int8_t* b_panel, int8_t* sme_a, TOut* fdst, const float* sb_ptr,
            const float* col_bias_f)
{
#if defined(NNOPS_ARCH_SME)
    if (c.use_sme) {
        if (!a.packed) {
            tile_pack_lhs_i8_sme(false, mc, kdim4, sme_a, a.ptr, a.lda, c.a_biased);
        }
        if constexpr (I8ConvCtx<TAct, TOut>::kIntOut) {
            if (c.epi_i8) {
                tile_mma_pack_i8_sme_epi_i8(
                    mc, static_cast<int>(ocnc), kdim4, fdst, static_cast<int64_t>(c.OC),
                    sme_a, b_panel, c.sme.slot_max, sb_ptr, c.qw->scale, c.s_x / c.out_scale,
                    c.b_i32 ? c.b_i32 + g * c.oc_per_group + ocn : nullptr, c.out_zp,
                    c.attrs->epilogue.type == EpilogueActivateType::Relu);
                return true;
            }
        } else if (c.epi_fused) {
            // The dequantizing store: the epilogue is the kernel's write-back,
            // so no int32 tile exists.
            tile_mma_pack_i8_sme_epi(mc, static_cast<int>(ocnc), kdim4, fdst,
                                     static_cast<int64_t>(c.OC), sme_a, b_panel,
                                     c.sme.slot_max, sb_ptr, c.qw->scale, c.s_x, col_bias_f,
                                     c.epi_lo, c.epi_hi);
            return true;
        }
        tile_mma_pack_i8_sme(mc, static_cast<int>(ocnc), kdim4, acc_tile,
                             static_cast<int>(c.ocn_block), sme_a, b_panel, c.sme.slot_max,
                             INT32_MIN, INT32_MAX, /*zero_mode=*/icn == 0);
        return false;
    }
#endif
    const int64_t ldd_a = geom_ldd_i8_a(kdim4);
    tile_pack_lhs_i8(false, mc, kdim4, a_pack, static_cast<int>(ldd_a), a.ptr, a.lda,
                     c.a_biased);
    tile_mma_pack_i8(mc, static_cast<int>(ocnc), kdim4, acc_tile,
                     static_cast<int>(c.ocn_block), a_pack, b_panel, /*ldb=*/0, INT32_MIN,
                     INT32_MAX, /*zero_mode=*/icn == 0);
    return false;
}

/// The per-pixel sums the compensation reads: over the whole K, of the *true*
/// activation values (the gather's shift comes back out), recomputed rather
/// than buffered. Returns nullptr when no compensation is needed.
template <class TAct, class TOut>
const int32_t* compute_i8_ra(const I8ConvCtx<TAct, TOut>& c, const uint8_t* in_g,
                             const uint8_t* in_row, int64_t ow0, int64_t p, int mc, bool flat,
                             bool direct_a, int8_t* a_buf, int32_t* ra_tile)
{
    if (!c.comp) {
        return nullptr;
    }
    if (direct_a) {
        // The pixel is its own patch, its channels one contiguous run: a row
        // sum of the bytes the MMA reads — no gather, no A tile.
        for (int m = 0; m < mc; ++m) {
            ra_tile[m] = k::int8_row_sum<int8_t>(
                reinterpret_cast<const int8_t*>(in_row) +
                    static_cast<int64_t>(ow0 + m) * c.IC,
                static_cast<int>(c.ic_per_group));
        }
    } else {
        for (int m = 0; m < mc; ++m) {
            ra_tile[m] = 0;
        }
        for (int64_t icn2 = 0; icn2 < c.ic_per_group; icn2 += c.icn_block) {
            const int64_t icnc2 = std::min(c.icn_block, c.ic_per_group - icn2);
            const int kd2 = static_cast<int>(icnc2 * c.karea);
            const int kd24 = static_cast<int>(align4(kd2));
            tiled_im2col_i8(a_buf, kd24, in_g + icn2, c.in_row_stride, c.IC, c.xr,
                            static_cast<uint8_t>(c.plane ? c.zp_x + c.d : c.zp_x), p,
                            static_cast<int>(c.OW), flat, mc, static_cast<int>(c.IH),
                            static_cast<int>(c.IW), c.KH, c.KW, c.SH, c.SW, c.DH, c.DW, c.PH,
                            c.PW, static_cast<int>(icnc2), c.karea, kd2, kd24);
            for (int m = 0; m < mc; ++m) {
                ra_tile[m] +=
                    k::int8_row_sum<int8_t>(a_buf + static_cast<int64_t>(m) * kd24, kd2);
            }
        }
    }
    const int32_t shift_total = c.d * static_cast<int32_t>(c.k_full);
    for (int m = 0; m < mc; ++m) {
        ra_tile[m] -= shift_total;
    }
    return ra_tile;
}

/// Compensate and dequantize (or requantize) one accumulator tile.
template <class TAct, class TOut>
void i8_epilogue(const I8ConvCtx<TAct, TOut>& c, int64_t g, int64_t ocn, int64_t ocnc,
                 int mc, int32_t* acc_tile, const int32_t* ra_ptr, TOut* dst,
                 const TOut* bias_g, const int32_t* ibias_g, bool epi_done)
{
    if (epi_done) {
        return;
    }
    const int64_t off = g * c.oc_per_group + ocn;
    const float* const s_b = c.qw->scale_data ? c.qw->scale_data + off : nullptr;
    const int32_t* const zp_b =
        c.qw->zero_point_data ? c.qw->zero_point_data + off : nullptr;
    const int32_t* const r_b = c.col_term ? c.r_b_all.data() + off : nullptr;

    if constexpr (I8ConvCtx<TAct, TOut>::kIntOut) {
        Int8EpilogueParams ip{};
        ip.r_a = ra_ptr;
        ip.s_a = nullptr; // per-tensor activation
        ip.zp_a = nullptr;
        ip.r_b = r_b;
        ip.s_b = s_b;
        ip.zp_b = zp_b;
        ip.s_a_scalar = c.s_x;
        ip.s_b_scalar = c.qw->scale;
        ip.zp_a_scalar = c.zp_x;
        ip.zp_b_scalar = c.qw->zero_point;
        ip.a_bias_offset = c.d;
        ip.k = static_cast<int32_t>(c.k_full);
        ip.relu = c.attrs->epilogue.type == EpilogueActivateType::Relu;
        ip.compensate = c.comp;
        ip.col_term = c.col_term;
        ip.out_scale = quant_scale_at(*c.qc, 0);
        ip.zp_out = quant_zp_at(*c.qc, 0);
        if (c.attrs->epilogue.quant == EpilogueQuantType::Requantize &&
            c.attrs->epilogue.quant_param_count > 0) {
            ip.row_out_scales = c.attrs->epilogue.quant_scales;
            ip.row_out_zps = c.attrs->epilogue.quant_zero_points;
            ip.out_scale = epilogue_quant_scale(c.attrs->epilogue, 0);
            ip.zp_out = epilogue_quant_zero(c.attrs->epilogue, 0);
        }
        ip.bias = nullptr; // the bias is per column
        ip.col_bias = ibias_g ? ibias_g + ocn : nullptr;
        if constexpr (std::is_same_v<TOut, uint8_t>) {
            ip.out_u8 = dst;
        } else {
            ip.out = dst;
        }
        ip.ldc_out = static_cast<int>(c.OC);
        int8_epilogue_tile(ip, acc_tile, static_cast<int>(c.ocn_block), 0, 0, mc,
                           static_cast<int>(ocnc));
    } else {
        DqEpilogueParams<TOut> ip{};
        ip.r_a = ra_ptr;
        ip.s_a = nullptr;
        ip.zp_a = nullptr;
        ip.r_b = r_b;
        ip.s_b = s_b;
        ip.zp_b = zp_b;
        ip.row_off = nullptr;
        ip.s_a_scalar = c.s_x;
        ip.s_b_scalar = c.qw->scale;
        ip.zp_a_scalar = c.zp_x;
        ip.zp_b_scalar = c.qw->zero_point;
        ip.a_bias_offset = c.d;
        ip.k = static_cast<int32_t>(c.k_full);
        ip.compensate = c.comp;
        ip.col_term = c.col_term;
        ip.act_u8 = c.u8a;
        ip.ep = &c.attrs->epilogue;
        ip.bias = nullptr;
        ip.col_bias = bias_g ? bias_g + ocn : nullptr;
        ip.out = dst;
        ip.ldc_out = static_cast<int>(c.OC);
        dq_epilogue_tile<TOut>(ip, acc_tile, static_cast<int>(c.ocn_block), 0, 0, mc,
                               static_cast<int>(ocnc));
    }
}

/// One task of the (batch, group) × oh-chunk × oc-chunk grid.
template <class TAct, class TOut> void run_i8_task(const I8ConvCtx<TAct, TOut>& c, int64_t t)
{
    const int64_t chunks_per_ng = c.oh_chunks * c.oc_chunks;
    const int64_t ng = t / chunks_per_ng;
    const int64_t rem = t % chunks_per_ng;
    const int64_t chunk = rem / c.oc_chunks;
    const int64_t oc_chunk = rem % c.oc_chunks;
    const int64_t n = ng / c.G;
    const int64_t g = ng % c.G;

    const int64_t slot =
        c.per_thread ? static_cast<int64_t>(c.ctx->cpu.current_thread_id()) : t;
    char* const slot_ptr = c.base + c.sc.head + slot * c.sc.slot_stride;
    int8_t* const a_pack = reinterpret_cast<int8_t*>(slot_ptr + c.sc.off_a);
    int8_t* const a_buf = a_pack + c.a_pack_stride;
    int8_t* const stage = reinterpret_cast<int8_t*>(slot_ptr + c.sc.off_stage);
    int8_t* const pack_buf = reinterpret_cast<int8_t*>(slot_ptr + c.sc.off_pack);
    int32_t* const acc_buf = reinterpret_cast<int32_t*>(
        c.base + c.sc.off_acc + slot * c.acc_size * static_cast<int64_t>(sizeof(int32_t)));

    // The SME route's per-task A slab; B is the shared prepack.
    PoolArray<int8_t> sme_a;
#if defined(NNOPS_ARCH_SME)
    if (c.use_sme) {
        const int64_t kd4 = align4(c.kdim_max);
        const int64_t a_blocks = (c.m_block + aarch64::sme_mr - 1) / aarch64::sme_mr;
        sme_a.reset(static_cast<size_t>(a_blocks * sme_a_panel_stride(DataType::s8, kd4)));
    }
#endif

    TOut* const out_g = c.out_ptr + n * c.ob_step + g * c.oc_per_group;
    const uint8_t* const in_g = c.in_base + n * c.ib_step + g * c.ic_per_group;
    const TOut* const bias_g = c.b_ptr ? c.b_ptr + g * c.oc_per_group : nullptr;
    const int32_t* const ibias_g = c.b_i32 ? c.b_i32 + g * c.oc_per_group : nullptr;

    const int64_t r0 = chunk * c.OH / c.oh_chunks;
    const int64_t r1 = (chunk + 1) * c.OH / c.oh_chunks;
    // Over *panels*, like the fp route: a task starts where a panel does. A ZA
    // blob's blocks are the filter's own 64-column ones, not the NEON panels
    // the split is measured in; its plan keeps the group whole.
    const int64_t oc_lo =
        c.sme_blob ? 0 : oc_chunk * c.pg.panels_oc / c.oc_chunks * I8ConvCtx<TAct, TOut>::nr_max;
    const int64_t oc_hi =
        c.sme_blob ? c.oc_per_group :
                     std::min((oc_chunk + 1) * c.pg.panels_oc / c.oc_chunks *
                                  I8ConvCtx<TAct, TOut>::nr_max,
                              c.oc_per_group);

    // Same flat-pixel rule as the fp route; pointwise also needs the input
    // dense, since it reads A straight out of the row.
    const bool flat =
        c.out_row_stride == c.OW * c.OC && (!c.pointwise || c.in_row_stride == c.IW * c.IC);

    [[maybe_unused]] std::vector<float> bias_f;
    // Same block-end clamp and step as the fp route: the blob holds one oc
    // block at a time and the kernel walks its panels at `slot` apart.
    const auto block_end = [&](int64_t ocn) {
        return c.panels ? std::min(oc_hi, (ocn / c.ocn_block + 1) * c.ocn_block) : oc_hi;
    };
    for (int64_t ocn = oc_lo; ocn < oc_hi;
         ocn = c.panels ? block_end(ocn) : ocn + c.ocn_block) {
        const int64_t oc_end = block_end(ocn);
        const int64_t ocnc = std::min(c.ocn_block, oc_end - ocn);

        // The epilogue's per-column weight scale and bias, indexed like
        // `ip.s_b` / `ip.col_bias` in i8_epilogue.
        const float* const sb_ptr =
            c.qw->scale_data ? c.qw->scale_data + g * c.oc_per_group + ocn : nullptr;
        // The fused store takes the bias in the float domain; a half bias
        // (which the dequantizing pass widens per element) widens once here.
        const float* col_bias_f = nullptr;
#if defined(NNOPS_ARCH_SME)
        if (c.epi_fused && bias_g != nullptr) {
            if constexpr (std::is_same_v<TOut, float>) {
                col_bias_f = bias_g + ocn;
            } else {
                // Sized by the tile, not by kOcnBlock: the int8 route's N tile
                // is the whole group width.
                if (bias_f.size() < static_cast<size_t>(ocnc)) {
                    bias_f.resize(static_cast<size_t>(ocnc));
                }
                for (int64_t j = 0; j < ocnc; ++j) {
                    bias_f[j] = static_cast<float>(bias_g[ocn + j]);
                }
                col_bias_f = bias_f.data();
            }
        }
#endif

        for (int64_t pr = r0; pr < r1; pr += c.ph_block) {
            const int64_t pr1 = std::min(r1, pr + c.ph_block);
            const int64_t p_first = pr * c.OW;
            const int64_t p_last = pr1 * c.OW;
            // Re-split a tail shorter than MR evenly, keeping the step a
            // multiple of MR so the re-split costs no extra block.
            constexpr int64_t mr_step = MR_MAX_I8;
            const int64_t p_span = std::max<int64_t>(p_last - p_first, 1);
            const int64_t p_tail = p_span % c.m_block;
            const int64_t p_step =
                (flat && p_tail != 0 && p_tail < mr_step && c.m_block >= 2 * mr_step) ?
                    (c.m_block / mr_step - 1) * mr_step :
                    c.m_block;

            for (int64_t icn = 0; icn < c.ic_per_group; icn += c.icn_block) {
                const int64_t icnc = std::min(c.icn_block, c.ic_per_group - icn);
                const bool is_last_icn = (icn + c.icn_block) >= c.ic_per_group;
                const int kdim = static_cast<int>(icnc * c.karea);
                const int kdim4 = static_cast<int>(align4(kdim));
                const bool direct_a = c.pointwise && (kdim == kdim4);

                const int8_t* const b_panel =
                    pick_i8_b(c, g, icn, ocn, ocnc, icnc, kdim, oc_chunk, stage, pack_buf);

                // `mc` lives outside the loop: it is the step, and the body's
                // early exits still have to advance it.
                int mc = 0;
                for (int64_t p = p_first; p < p_last; p += flat ? p_step : mc) {
                    const int64_t oh = p / c.OW;
                    const int64_t ow0 = p - oh * c.OW;
                    const int64_t left = flat ? std::min(p_step, p_last - p) : (c.OW - ow0);
                    mc = static_cast<int>(std::min<int64_t>(c.m_block, left));
                    const uint8_t* const in_row = in_g + oh * c.in_row_stride;

                    const I8ATile a =
                        pick_i8_a(c, in_g, in_row, icn, icnc, p, ow0, mc, flat, kdim, kdim4,
                                  a_buf, sme_a.data());
                    int32_t* const acc_tile = acc_buf + (p - p_first) * c.ocn_block;
                    TOut* const dst = out_g + oh * c.out_row_stride + ow0 * c.OC + ocn;

                    const bool epi_done =
                        i8_mma(c, g, icn, mc, ocn, ocnc, kdim4, acc_tile, a_pack, a, b_panel,
                               sme_a.data(), dst, sb_ptr, col_bias_f);
                    if (!is_last_icn) {
                        continue;
                    }

                    int32_t ra_tile[kMaxMBlock];
                    const int32_t* const ra_ptr =
                        compute_i8_ra(c, in_g, in_row, ow0, p, mc, flat, direct_a, a_buf,
                                      ra_tile);
                    i8_epilogue(c, g, ocn, ocnc, mc, acc_tile, ra_ptr, dst, bias_g, ibias_g,
                                epi_done);
                }
            }
        }
    }
}

// ---- main kernel ----------------------------------------------------------

template <class TAct, class TOut>
void conv2d_int8_impl(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs, const ComputeContext& ctx,
                      const Conv2DGemmPlan& plan, void* workspace)
{
    I8ConvCtx<TAct, TOut> c;
    init_i8_ctx(c, attrs, output, inputs, ctx, plan, workspace);
    ctx.cpu.run(0, c.total, [&](int64_t t) { run_i8_task(c, t); });
}

// =========================================================================
//  Plan
// =========================================================================

/// The dtype ladder this route admits; false leaves the caller's plan zeroed
/// (the unsupported answer).
bool conv_plan_dtypes_ok(const Conv2DAttributes& attrs, std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs, bool quant) noexcept
{
    const DataType dt = inputs[0].dtype;
    const DataType wt_dt = inputs[1].dtype;
    const DataType out_dt = outputs[0].dtype;
    if (quant) {
        // Activation: pre-quantized (u8/s8), or fp (quantized per call).
        if (dt != DataType::u8 && dt != DataType::s8 && dt != DataType::f32 &&
            dt != DataType::f16) {
            return false;
        }
        const bool int_out = out_dt == DataType::s8 || out_dt == DataType::u8;
        if (!int_out && out_dt != DataType::f32 && out_dt != DataType::f16) {
            return false;
        }
        if (!int_out && (dt == DataType::f32 || dt == DataType::f16) && out_dt != dt) {
            return false;
        }
        const DataType bias_dt = int_out ? DataType::s32 : out_dt;
        if (inputs.size() > 2 && inputs[2].dtype != bias_dt) {
            return false;
        }
        // Only the reference implements add_to's fold on the quantized route.
        return !attrs.add_to;
    }
    if (!is_fp_activation_dtype(dt)) {
        return false;
    }
    if (wt_dt != dt || out_dt != dt) {
        return false;
    }
    return inputs.size() <= 2 || inputs[2].dtype == dt;
}

Conv2DGemmPlan make_plan(const Conv2DAttributes& attrs, std::span<const TensorDesc> inputs,
                         std::span<const TensorDesc> outputs, int num_threads,
                         bool use_thread_slots, int64_t num_slots)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(outputs.size() >= 1);

    const auto& in = inputs[0];
    const auto& wt = inputs[1];
    const auto& out = outputs[0];

    Conv2DGemmPlan plan;

    const DataType dt = in.dtype;
    const bool quant = is_quantized_conv2d(wt.dtype);
    if (!conv_plan_dtypes_ok(attrs, inputs, outputs, quant)) {
        return plan;
    }

    const int64_t N = in.dims[0];
    const int64_t IC = in.dims[3];
    // The weight is TensorFlow's [KH, KW, IC/G, OC]; anything else
    // is not this route's — except the prepacked blob, whose shape leads with OC
    // (see conv_packed_weight_desc).
    const bool sme_blob = wt.layout == TensorLayout::PackedHWIOSme;
    const bool prepacked = is_packed_conv_weight(wt.layout);
    if (!is_channel_last_weight(wt.layout) && !prepacked) {
        return plan;
    }
    const int64_t OC = prepacked ? wt.dims[0] : wt.dims[3];
    const int64_t OH = out.dims[1];
    const int64_t OW = out.dims[2];

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];

    // A 1x1 kernel at unit stride with no padding reads the input row itself as
    // A — no gather. No pitch test needed: a pixel's channels are contiguous
    // whatever the row pitch, and a tile never leaves its output row.
    plan.pointwise = (karea == 1 && attrs.stride[0] == 1 && attrs.stride[1] == 1 &&
                      attrs.padding[0] == 0 && attrs.padding[1] == 0);

    // The block sizes are the prepacked filter's geometry: conv_packed_weight
    // resolves them from the attributes and the filter's dims alone, and a plan
    // choosing again would size panels the blob does not hold.
    const ConvPackedWeight pg = sme_blob ?
                                    conv_packed_weight_sme(attrs, ic_per_group, oc_per_group,
                                                           wt.dtype) :
                                    conv_packed_weight(attrs, ic_per_group, oc_per_group, wt.dtype);
    plan.icn_block = pg.icn_block;
    plan.ocn_block = pg.ocn_block;
    const int64_t nr_for_split = pg.nr;
#if defined(NNOPS_ARCH_SME)
    // The SME route's tile is two 16-row ZA faces: 32 pixels. A 24-row kMBlock
    // would leave a quarter of every tile's rows dead and take the masked
    // fallback instead of the pair load.
    const bool sme_route = !quant &&
                           (sme_blob || (!prepacked && aarch64::available_sme() &&
                                         oc_per_group >= aarch64::sme_fp_nr));
    // The int8 route's A tile is 256 pixels whatever the row width. Width was
    // once the gate here, on the theory that a narrow row cannot fill the tile;
    // measured, the opposite holds — the tile is what the kernel wants, and a
    // 24-pixel one (the NEON kMBlock) re-walks the whole packed B every 24
    // pixels. 28x28 128->128 ran 128 us against 104 us at 128 pixels; 256
    // gives another 1-2% on top by halving the calls again.
    const bool sme_i8_route = quant &&
                              (sme_blob || (!prepacked && aarch64::available_sme() &&
                                            oc_per_group >= aarch64::sme_nr_max));
    static_assert(16 * aarch64::sme_mr <= kMaxMBlock, "kMaxMBlock must cover the int8 tile");
    plan.m_block = sme_route ? 2 * aarch64::sme_mr : (sme_i8_route ? 16 * aarch64::sme_mr : kMBlock);
#else
    plan.m_block = kMBlock;
#endif
    plan.oh_chunks = 1;
    plan.oc_chunks = 1;

    const int64_t NG = N * G;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    const int64_t esz = quant ? 1 : static_cast<int64_t>(data_type_size(dt));

    // Rows first (the axis every task shares), then the filter axis when the
    // rows cannot fill the pool — with the nearest-round rule.
    //
    // A row chunk must still be worth a micro-kernel tile: split a 7x7 plane to
    // one row and each task is seven pixels against a 24-row tile. So the row
    // split stops at the chunk count that keeps `kTaskMinPixels` in each.
    const int64_t ow_at_least = std::max<int64_t>(OW, 1);
    const int64_t rows_min = std::max<int64_t>(1, (kTaskMinPixels + ow_at_least - 1) / ow_at_least);
    const int64_t max_row_chunks = std::max<int64_t>(1, OH / rows_min);
    plan.oh_chunks = std::clamp<int64_t>((nt + NG - 1) / NG, 1,
                                         std::min(std::max<int64_t>(OH, 1), max_row_chunks));
    const int64_t plane_tasks = NG * plan.oh_chunks;
    // An SME blob's OC blocks are 64-aligned; its tasks keep one whole group on
    // the filter axis rather than starting mid-block.
    if (!sme_blob && 2 * plane_tasks < nt) {
        // The filter split divides N, so its granularity is a panel (`nr`): two
        // panels per chunk is the smallest split that fills a micro-kernel tile.
        // Splitting 96 oc seven ways leaves 14 live lanes of a 16-wide panel.
        const int64_t max_chunks = std::max<int64_t>(1, oc_per_group / (2 * nr_for_split));
        const int64_t want = (nt + plane_tasks / 2) / plane_tasks;
        plan.oc_chunks = std::clamp<int64_t>(want, 1, max_chunks);
    }

    const bool has_bias = inputs.size() > 2;
    const bool epi_active = attrs.epilogue.type != EpilogueActivateType::None;
    const bool epi_folded = !attrs.add_to && !has_bias &&
                            (attrs.epilogue.type == EpilogueActivateType::Relu ||
                             attrs.epilogue.type == EpilogueActivateType::Relu6);
    const bool keep_orig = !quant && attrs.add_to && epi_active && !epi_folded;

    // The residual tile set must cover the task's whole row range for one oc
    // block; split the rows further until it fits (parallelism, not memory).
    if (keep_orig) {
        while (plan.oh_chunks < OH &&
               ((OH + plan.oh_chunks - 1) / plan.oh_chunks) * OW * plan.ocn_block > kTaskTileCap) {
            ++plan.oh_chunks;
        }
    }

    const int64_t rows_in_task = (OH + plan.oh_chunks - 1) / plan.oh_chunks;

    // Rows per accumulator pass. The accumulator is re-read and written once per
    // ic block over the rows one pass covers, so that set — not the whole task —
    // must stay in cache: a 56×56 full-tile set is 800 KB and every pass went to
    // DRAM. The shared prepack makes the extra passes free; a per-task pack pays
    // one pack per (oc block, pass, ic block).
    const int64_t pass_esz = std::max<int64_t>(4, esz);
    plan.ph_block =
        std::clamp<int64_t>(kPassAccBytes / (pass_esz * plan.ocn_block * std::max<int64_t>(OW, 1)),
                            1, std::max<int64_t>(rows_in_task, 1));

    const int64_t nr_max = pg.nr;
    const int64_t kdim_max = pg.kdim_max;
    const int64_t panels_per_oc = pg.panels_oc;

    // The panels: the caller's blob, or one store per call written by this
    // route's own prepack in the identical layout — so a raw and a prepacked
    // call address them the same way, and the filter-axis split needs no
    // special case. Past the cap, one panel set per slot, packed per task.
    const bool shared = !prepacked && pg.total * esz <= kPrepackCap;

    plan.shared_elems = shared ? pg.total : 0;
    plan.pack_size = (prepacked || shared) ? 0 : panels_per_oc * pg.slot_full;
    // int8 gathers bytes padded to whole 4-byte groups (the direct-A kernel reads
    // kc four at a time) — which is why a pointwise int8 tile whose channel block
    // is not a multiple of 4 still needs the gather buffer: the in-row A read
    // cannot be zero-padded.
    const bool pw_gather = quant && plan.pointwise && (ic_per_group % 4 != 0);
    // The quantized route hands the MMA a *packed* A panel (direct-A costs
    // ~2.7x the uops). The panel region comes first in the slot's A area, so its
    // stride is the gather's offset.
    plan.a_pack_stride =
        quant ? gemm_align64(
                    static_cast<int64_t>(num_panels_max(static_cast<int>(plan.m_block), MR_I8)) *
                    geom_ldd_i8_a((kdim_max + 3) & ~int64_t{3})) :
                0;
    plan.a_size = plan.a_pack_stride + ((plan.pointwise && !pw_gather) ?
                                            0 :
                                            plan.m_block * (quant ? ((kdim_max + 3) & ~int64_t{3}) :
                                                                    kdim_max + kABSlack));
    // In HWIO a panel's k row is one OC-length run, nothing transposed. A 1x1
    // filter is read in place and needs only the tail's panel; a multi-tap one
    // restages a whole panel. Either way, only when a task does pack — which
    // neither a prepacked filter nor a shared per-call one does.
    plan.stage_size = plan.pack_size > 0 ? nr_max * kdim_max : 0;
    const int64_t pass_tiles = plan.ph_block * OW;
    plan.orig_size = keep_orig ? pass_tiles * plan.ocn_block : 0;
    // The int32 accumulator is a per-slot tile set shaped like the residual.
    plan.acc_size = quant ? pass_tiles * plan.ocn_block : 0;
    // An fp activation is materialized into a compact s8 NHWC plane once per
    // call. A pre-quantized u8 one is materialized too when it takes the
    // pointwise route (whose in-row A read cannot shift bytes) or the SME one,
    // whose gather packs A straight out of ZA rows and so cannot shift either.
#if defined(NNOPS_ARCH_SME)
    const bool sme_i8_plane = !prepacked && aarch64::available_sme() &&
                              oc_per_group >= aarch64::sme_nr_max;
#else
    constexpr bool sme_i8_plane = false;
#endif
    const bool fp_act = (dt == DataType::f32 || dt == DataType::f16);
    plan.quant_size = (quant && (fp_act || (dt == DataType::u8 &&
                                            (plan.pointwise || sme_blob || sme_i8_plane)))) ?
                          N * in.dims[1] * in.dims[2] * IC :
                          0;

    const int64_t total = NG * plan.oh_chunks * plan.oc_chunks;
    const int64_t avail = (num_slots > 0) ? num_slots : nt;
    plan.num_slots = (use_thread_slots && avail < total) ? avail : total;

    plan.workspace_size = static_cast<size_t>(gemm_scratch_layout(plan, esz).total);
    return plan;
}

} // anonymous namespace
} // namespace impl

// =========================================================================
//  Public API — one route per axis, chosen from the activation's layout
// =========================================================================

namespace {

/// A route is (make_plan, fp_impl, quant_impl): the plan, then the two per-dtype
/// leaves the shared ladders below call. Everything route-independent — the
/// pool, the plan prologue, the column split, the dtype switches — lives outside.

struct GemmRoute {
    static Conv2DGemmPlan make_plan(const Conv2DAttributes& attrs,
                                    std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs, int num_threads,
                                    bool use_thread_slots, int64_t num_slots)
    {
        return impl::make_plan(attrs, inputs, outputs, num_threads, use_thread_slots,
                                       num_slots);
    }

    template <class T>
    static void fp_impl(const Conv2DAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx,
                        const Conv2DGemmPlan& plan, void* workspace)
    {
        impl::conv2d_impl<T>(attrs, output, inputs, ctx, plan, workspace);
    }

    /// The storage choice is a runtime flag inside this route's kernel, not a
    /// template parameter: its packer reads both signs on one path.
    template <class TAct, class TOut>
    static void quant_impl(const Conv2DAttributes& attrs, TensorView& output,
                           std::span<const TensorView> inputs, const ComputeContext& ctx,
                           const Conv2DGemmPlan& plan, void* workspace)
    {
        impl::conv2d_int8_impl<TAct, TOut>(attrs, output, inputs, ctx, plan,
                                                                workspace);
    }
};

// ---- dtype ladders --------------------------------------------------------
//
// Route-independent, so each is written once and takes its route as a parameter:
// the fp ladder switches on the activation's element type, the quantized one on
// (activation, output). The weight chose the family in conv2d_gemm_run, and the
// plan refused every combination these would miss — a refusal here means the
// plan and the ladders disagree, not that the caller asked for something odd.

template <class TRoute, class TAct>
Status conv2d_quant_out(const Conv2DAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx,
                        const Conv2DGemmPlan& plan, void* workspace)
{
    switch (output.data_type()) {
    case DataType::f32:
        TRoute::template quant_impl<TAct, float>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    case DataType::f16:
        TRoute::template quant_impl<TAct, half>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    case DataType::s8:
        TRoute::template quant_impl<TAct, int8_t>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    case DataType::u8:
        TRoute::template quant_impl<TAct, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

template <class TRoute>
Status conv2d_quant_run(const Conv2DAttributes& attrs, TensorView& output,
                        std::span<const TensorView> inputs, const ComputeContext& ctx,
                        const Conv2DGemmPlan& plan, void* workspace)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        return conv2d_quant_out<TRoute, float>(attrs, output, inputs, ctx, plan, workspace);
    case DataType::f16:
        return conv2d_quant_out<TRoute, half>(attrs, output, inputs, ctx, plan, workspace);
    case DataType::u8:
        return conv2d_quant_out<TRoute, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
    case DataType::s8:
        return conv2d_quant_out<TRoute, int8_t>(attrs, output, inputs, ctx, plan, workspace);
    default:
        return Status::unsupported;
    }
}

template <class TRoute>
Status conv2d_fp_run(const Conv2DAttributes& attrs, TensorView& output,
                     std::span<const TensorView> inputs, const ComputeContext& ctx,
                     const Conv2DGemmPlan& plan, void* workspace)
{
    switch (inputs[0].data_type()) {
    case DataType::f32:
        TRoute::template fp_impl<float>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    case DataType::f16:
        TRoute::template fp_impl<half>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16:
        TRoute::template fp_impl<bfloat16>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
#endif
    default:
        return Status::unsupported;
    }
}

/// [N, H, W, C] cut to `len` columns from `col0`, pitch and layout kept: a
/// column is a whole channel vector.
TensorView wide_window(const TensorView& v, int64_t col0, int64_t len)
{
    const int64_t unit = static_cast<int64_t>(data_type_stride_unit(v.data_type()));
    const int64_t off = col0 * v.shape(3) * unit;
    const int64_t s[4] = {v.shape(0), v.shape(1), len, v.shape(3)};
    auto* base =
        const_cast<char*>(static_cast<const char*>(static_cast<const void*>(v.ptr<uint8_t>())));
    return TensorView(std::span<const int64_t>(s, 4), v.data_type(), base + off, v.pitch(),
                      v.layout(), v.quant_params());
}

/// A one-row output has no oh axis to split, so the plan is left with the filter
/// — and a filter chunk still needs every activation: it duplicates the im2col
/// gather once per chunk where a column chunk divides it. MLAS picks between M
/// and N for the same reason (the larger one), and for a one-row image N = the
/// row is normally the larger. Returns the column-chunk count, 1 for whole.
int64_t wide_split_chunks(const Conv2DAttributes& attrs, std::span<const TensorView> inputs,
                          const TensorView& output, const Conv2DGemmPlan& plan,
                          const ComputeContext& ctx)
{
    const TensorView& input = inputs[0];
    const TensorView& weight = inputs[1];
    const int64_t OW = output.shape(2);
    const int64_t IW = input.shape(2);
    const int64_t workers = std::max<int64_t>(ctx.cpu.worker_count(), 1);
    if (output.shape(1) != 1 || workers <= 1 || inputs.size() > 4) {
        return 1;
    }

    // A chunk pays the filter's halo again, so eight effective kernels of output
    // amortize it. The quantized route also repacks the whole filter into the
    // MMA's A panels per task — compute, not traffic, invisible to the bound
    // below — so its chunks stay longer.
    const int64_t k_eff = attrs.dilation[1] * (attrs.kernel_size[1] - 1) + 1;
    const int64_t min_len =
        std::max<int64_t>(8 * k_eff, is_quantized_conv2d(weight.data_type()) ? 128 : 1);
    if (OW < 2 * min_len || (OW - 1) * attrs.stride[1] - attrs.padding[1] >= IW) {
        return 1;
    }

    // 1 + 8·io/filter: past that the chunks bury the compute in filter streams.
    const int64_t filter_bytes =
        weight.numel() * static_cast<int64_t>(data_type_size(weight.data_type()));
    const int64_t io_bytes =
        input.numel() * static_cast<int64_t>(data_type_size(input.data_type())) +
        output.numel() * static_cast<int64_t>(data_type_size(output.data_type()));
    const int64_t by_traffic = 1 + (8 * io_bytes) / std::max<int64_t>(filter_bytes, 1);

    // Batch, group and oh divide the gather; only oc_chunks buys width by
    // duplicating it.
    const int64_t plane_items = input.shape(0) * attrs.groups * plan.oh_chunks;
    const int64_t tasks = plane_items * plan.oc_chunks;
    const int64_t chunks = std::clamp<int64_t>(std::min(workers, OW / min_len), 1, by_traffic);

    // Take the column axis only when it is at least as wide as the plan's own
    // grid, and only where that grid is thin: once batch and group fill the
    // pool the split adds no parallelism, only the halo and the filter re-read
    // (a batch-8 layer at 4T read 0.79x with it).
    return (chunks < std::min(workers, tasks) || plane_items >= workers) ? 1 : chunks;
}

/// The shared prologue: descriptors from the views, one plan, one pooled arena,
/// then the route's driver. A zeroed plan (icn_block 0) is the "unsupported
/// dtype combination" answer, refused before anything is allocated.
template <class TRoute>
Status conv2d_gemm_run(const Conv2DAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    std::vector<TensorDesc> descs;
    descs.reserve(inputs.size());
    for (const auto& t : inputs) {
        descs.push_back(t.desc());
    }
    const TensorDesc outs[] = {output.desc()};

    const Conv2DGemmPlan plan =
        TRoute::make_plan(attrs, descs, outs, ctx.cpu.thread_count(),
                          ctx.cpu.thread_slot_count() > 0, ctx.cpu.thread_slot_count());
    if (plan.icn_block <= 0) {
        return Status::unsupported;
    }

    // The column split: each chunk is one serial call on its own output window,
    // so no nested region is opened, and the chunks write disjoint columns —
    // bit-exact whatever the count.
    const int64_t col_chunks = wide_split_chunks(attrs, inputs, output, plan, ctx);
    if (col_chunks > 1) {
        const int64_t OW = output.shape(2);
        const int64_t IW = inputs[0].shape(2);
        const int64_t S = attrs.stride[1];
        const int64_t P = attrs.padding[1];
        std::atomic<int> fail{0};
        ctx.cpu.run(0, col_chunks, [&](int64_t c) {
            const int64_t o0 = c * OW / col_chunks;
            const int64_t o1 = (c + 1) * OW / col_chunks;
            const int64_t origin = std::max<int64_t>(0, o0 * S - P);
            Conv2DAttributes ac = attrs;
            ac.padding[1] = std::max<int64_t>(0, P - o0 * S);
            TensorView ops[4];
            std::copy(inputs.begin(), inputs.end(), ops);
            ops[0] = wide_window(inputs[0], origin, IW - origin);
            TensorView out_c = wide_window(output, o0, o1 - o0);
            const Status st = conv2d_implicit_gemm_kernel(
                ac, out_c, std::span<const TensorView>(ops, inputs.size()), ComputeContext{},
                nullptr);
            if (!is_ok(st)) {
                fail.store(static_cast<int>(st), std::memory_order_relaxed);
            }
        });
        const auto r = static_cast<Status>(fail.load());
        return is_ok(r) ? Status::ok : r;
    }

    // A pointwise call with no add_to needs no scratch, but the pool rejects a
    // zero-byte request: ask for the smallest arena instead.
    PoolPtr scratch(plan.workspace_size ? plan.workspace_size : 1);
    if (!scratch) {
        return Status::out_of_memory;
    }

    // The weight's dtype is the family, whatever the activation is: an s8 filter
    // selects the quantized kernels, and a pre-quantized activation does not.
    return is_quantized_conv2d(inputs[1].data_type()) ?
               conv2d_quant_run<TRoute>(attrs, output, inputs, ctx, plan, scratch.get()) :
               conv2d_fp_run<TRoute>(attrs, output, inputs, ctx, plan, scratch.get());
}

} // anonymous namespace

// =========================================================================
//  The prepacked filter (Conv2D::prepackWeights, im2col route)
// =========================================================================
//
// The pack compute() otherwise does is a function of the filter alone, so it is
// a load-time artifact: query a descriptor, allocate, fill once — the same
// two-step contract as the Winograd U. It is safe because the geometry comes from
// the attributes plus the filter's own dims, never the activation (hence these
// take channel counts, not a plan), and the plan adopts icn_block/ocn_block from
// conv_packed_weight rather than choosing again — so "the panels
// prepackWeights built" and "the panels the plan sized for" are one decision.
//
// The packed blob is addressed by *panel*: panel p holds output channels
// [p·nr, (p+1)·nr) of one k block, for every k block of the group — so any
// panel-aligned oc range is readable and the plan's N tile stops constraining it.

ConvPackedWeight conv_packed_weight(const Conv2DAttributes& attrs, int64_t ic_per_group,
                                             int64_t oc_per_group, DataType weight_dtype)
{
    ConvPackedWeight g;
    if (ic_per_group <= 0 || oc_per_group <= 0 || attrs.groups <= 0) {
        return g;
    }

    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];
    const bool quant = is_quantized_conv2d(weight_dtype);
    const int64_t esz = quant ? 1 : static_cast<int64_t>(data_type_size(weight_dtype));

    // Block sizes factorised here so the plan has one place to read them from and
    // prepackWeights can resolve them without an activation. A 1x1 filter at unit
    // stride runs K over the whole channel count in one pass; padding is not
    // consulted (no input shape), and a padded 1x1 gathers with the same wide K.
    const bool unit_1x1 = karea == 1 && attrs.stride[0] == 1 && attrs.stride[1] == 1;
    g.nr = quant ? NR_MAX_I8 : nr_max_dt(weight_dtype);
    // At least 32 input channels per k block: with a raw filter this geometry
    // is also the plan the ZA route packs its per-call B slab against, and a
    // shorter block (16 channels of a 3x3 is 144) leaves the tile's K loop too
    // short to amortise the per-call store and setup — 1050 GFLOPS at 144
    // against 1560 at 576.
    constexpr int64_t kSmeIcMin = 32;
    // The quantized route takes the group's whole K as well. Its side of the
    // trade is the gather: the nine-tap addressing is per pixel and per tap, so
    // a 32-channel block pays it once per block -- eight times for a 256-channel
    // 3x3. Whole-K measured 15% off all six ResNet18 shapes.
    g.icn_block =
        (unit_1x1 || quant) ? ic_per_group :
                   std::min<int64_t>(std::max<int64_t>(
                                         impl::get_ic_block(karea, ic_per_group), kSmeIcMin),
                                     ic_per_group);
    // Balance the oc blocks rather than taking kOcnBlock and leaving a sliver:
    // 144 is 80 + 64, not 128 + 16 (an eighth-full block paying every per-block
    // cost: accumulator fill, A pack, epilogue).
    const int64_t oc_blocks_wanted =
        oc_per_group <= impl::kOcnSingleCap ?
            1 :
            (oc_per_group + impl::kOcnBlock - 1) / impl::kOcnBlock;
    const int64_t per_block = (oc_per_group + oc_blocks_wanted - 1) / oc_blocks_wanted;
    g.ocn_block = std::min<int64_t>(oc_per_group, ((per_block + g.nr - 1) / g.nr) * g.nr);

    g.kdim_max = g.icn_block * karea;
    const bool bf16 = weight_dtype == DataType::bf16;
    g.slot_full = gemm_panel_slot(g.kdim_max, g.nr, esz, quant, bf16);
    g.nfull_ic = ic_per_group / g.icn_block;
    g.tail_ic = ic_per_group - g.nfull_ic * g.icn_block;
    g.slot_tail = g.tail_ic ? gemm_panel_slot(g.tail_ic * karea, g.nr, esz, quant, bf16) : 0;
    g.col_elems = g.nfull_ic * g.slot_full + (g.tail_ic ? g.slot_tail : 0);
    // ⌈⌉, not ÷: a lone oc block is the group's own width, which need not be a
    // multiple of nr (24 channels is two panels, one of them half live).
    g.block_panels = (g.ocn_block + g.nr - 1) / g.nr;
    // Count the group's panels the way the blob addresses them — one whole
    // block_panels column per oc block — not as ⌈oc_per_group/nr⌉. The two
    // differ whenever ocn_block's rounding up to nr leaves the last block
    // part-full: 512 channels on aarch64 are 2 blocks of 264, i.e. 44 panels,
    // where ⌈512/12⌉ = 43. Sizing by the blocks is what keeps the last block's
    // shorter panel column inside the buffer.
    const int64_t num_oc_blocks = (oc_per_group + g.ocn_block - 1) / g.ocn_block;
    g.panels_oc = num_oc_blocks * g.block_panels;
    g.group_elems = g.panels_oc * g.col_elems;
    g.total = attrs.groups * g.group_elems;
    if (quant) {
        // The R_b table trails the panels: int32 per output channel, the only
        // part of the blob that is not panel-formatted.
        g.sums_off = g.total;
        const int64_t bytes = attrs.groups * oc_per_group * static_cast<int64_t>(sizeof(int32_t));
        g.total += (bytes + 3) & ~int64_t{3};
    }
    return g;
}

ConvPackedWeight conv_packed_weight_sme(const Conv2DAttributes& attrs, int64_t ic_per_group,
                                        int64_t oc_per_group, DataType weight_dtype)
{
    ConvPackedWeight g;
    const bool q8 = weight_dtype == DataType::s8;
    if (ic_per_group <= 0 || oc_per_group <= 0 || attrs.groups <= 0 ||
        (!q8 && weight_dtype != DataType::f32 && weight_dtype != DataType::f16)) {
        return g;
    }

    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];
    const bool unit_1x1 = karea == 1 && attrs.stride[0] == 1 && attrs.stride[1] == 1;
    // The ZA n tile: 32 columns for the float 2x2 face pair, 64 for the int8
    // SMOPA kernel's four 16-column groups. Constants here so this geometry
    // stays buildable without the SME headers.
    constexpr int64_t kSmeBlockN_f32 = 32;
    constexpr int64_t kSmeBlockN_i8 = 64;
    g.sme = true;
    g.nr = q8 ? kSmeBlockN_i8 : kSmeBlockN_f32;
    // The ZA tiles cover 32x32, so a short ic block leaves the kernel's K loop
    // (kdim = icn * karea groups of four) too short to amortise the per-call
    // store and setup: 16 channels of a 3x3 is 144, and the tile ran at
    // 1050 GFLOPS against 1560 at 576.
    constexpr int64_t kSmeIcMin = 32;
    // int8 runs the group's whole K in one pass: the SMOPA tile wants a long K
    // loop, and every k block is another gather, another A pack and another
    // read-modify-write of the whole accumulator tile set.
    g.icn_block =
        (unit_1x1 || q8) ?
            ic_per_group :
            std::min<int64_t>(
                std::max<int64_t>(impl::get_ic_block(karea, ic_per_group), kSmeIcMin),
                ic_per_group);
    // The task N tile is a whole number of 32-column ZA blocks. The int8 route
    // takes the group's whole width: its A panel is gathered once per task and
    // every extra oc block re-gathers the same tile, so a 512-channel group
    // packed as four 128-column blocks paid for the im2col four times.
    const int64_t oc_tile = q8 ? oc_per_group : std::min<int64_t>(oc_per_group, impl::kOcnBlock);
    g.ocn_block = ((oc_tile + g.nr - 1) / g.nr) * g.nr;
    g.kdim_max = g.icn_block * karea;
    g.nfull_ic = ic_per_group / g.icn_block;
    g.tail_ic = ic_per_group - g.nfull_ic * g.icn_block;
    g.num_ic_blocks = g.nfull_ic + (g.tail_ic ? 1 : 0);
    g.num_oc_blocks = (oc_per_group + g.nr - 1) / g.nr;
    g.sme_region = sme_b_panel_stride(weight_dtype, g.kdim_max);
    g.slot_full = g.sme_region;
    g.panels_oc = g.num_oc_blocks;
    g.total = attrs.groups * g.num_ic_blocks * g.num_oc_blocks * g.sme_region;
    return g;
}


namespace {

/// The blob's descriptor: [OC, ⌈elems / OC⌉] with an explicit pitch, so numel()
/// is exactly the buffer — how the two-step prepack's callers size their
/// allocation. The dims after [0] carry no meaning.
TensorDesc packed_weight_desc(int64_t oc_per_group, int64_t groups, int64_t elems, DataType dtype,
                              TensorLayout layout)
{
    TensorDesc d;
    if (elems <= 0 || oc_per_group <= 0) {
        return d;
    }
    const int64_t OC = oc_per_group * groups;

    d.rank = 2;
    d.dtype = dtype;
    d.layout = layout;
    d.dims.resize(2);
    d.dims[0] = OC;
    d.dims[1] = (elems + OC - 1) / OC;
    d.row_stride_elems = d.dims[1];
    return d;
}

} // anonymous namespace

TensorDesc conv_packed_weight_desc(const Conv2DAttributes& attrs, const TensorDesc& weight)
{
    // The filter's own dims give the channel counts: HWIO is [KH][KW][IC/G][OC].
    if (weight.rank != 4 || !is_channel_last_weight(weight.layout)) {
        return TensorDesc{};
    }
    const int64_t OC = weight.dims[3];
    const int64_t ICG = weight.dims[2];
    const int64_t G = attrs.groups;
    if (G <= 0 || OC % G != 0 || ICG <= 0) {
        return TensorDesc{};
    }

    // The ZA tiles only repay their blob when the oc block fills four 64-column
    // blocks; below that the NEON panels are the better packed form.
#if defined(NNOPS_ARCH_SME)
    // The ZA routes' own widths: the float 2x2 tile is 32 columns wide, the
    // int8 SMOPA tile 64.
    const bool sme_fp = (weight.dtype == DataType::f32 || weight.dtype == DataType::f16) &&
                        (OC / G) >= aarch64::sme_fp_nr;
    const bool sme_q8 = weight.dtype == DataType::s8 && (OC / G) >= aarch64::sme_nr_max;
    const bool sme_blob = (sme_fp || sme_q8) && aarch64::available_sme();
#else
    const bool sme_blob = false;
#endif
    const ConvPackedWeight g = sme_blob ?
                                   conv_packed_weight_sme(attrs, ICG, OC / G, weight.dtype) :
                                   conv_packed_weight(attrs, ICG, OC / G, weight.dtype);
    return packed_weight_desc(OC / G, G, g.total, weight.dtype,
                              sme_blob ? TensorLayout::PackedHWIOSme :
                                         TensorLayout::PackedHWIO);
}

namespace {

/// Run @p body over [0, n) with one staging tile per worker — the NHWC
/// pack's only scratch. Falls back to one tile and a serial loop when the
/// backend reports no thread ids: a shared tile would race.
template <class Body>
void with_stage(const ComputeContext& ctx, int64_t n, int64_t stage_len, Body body)
{
    const int nslots = std::max(ctx.cpu.thread_slot_count(), 1);
    const bool par = nslots > 1 && n > 1;
    // One char array keeps this helper type-agnostic.
    PoolArray<char> stages(static_cast<size_t>(nslots * stage_len));
    auto run = [&](int64_t s) {
        char* const stage =
            stages.data() +
            (par ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : 0) * stage_len;
        body(s, stage);
    };
    if (par) {
        ctx.cpu.run(0, n, run);
    } else {
        for (int64_t s = 0; s < n; ++s) {
            run(s);
        }
    }
}

/// Pack the HWIO panels (PackedHWIO): one panel per (group, oc block,
/// k block, panel), `nr` output channels wide.
///
/// A panel is what pack_block builds for an oc run one panel wide. The task
/// split is (group, oc block, k block) — the layout's own unit, so every task
/// writes a contiguous region without the panels of two oc blocks interleaving.
template <class T>
void pack_weight(const TensorView& weight, TensorView& packed, const ConvPackedWeight& g,
                      int64_t ic_per_group, int64_t oc_per_group, int64_t groups,
                      const ComputeContext& ctx)
{
    const int64_t karea = g.kdim_max / g.icn_block;
    const int64_t num_ic_blocks = g.nfull_ic + (g.tail_ic ? 1 : 0);
    const int64_t num_oc_blocks = (oc_per_group + g.ocn_block - 1) / g.ocn_block;
    const int64_t per_group = num_oc_blocks * num_ic_blocks;

    const T* const w = weight.ptr<T>();
    T* const blob = packed.ptr<T>();
    const int64_t stage_len = g.nr * g.kdim_max; // elements

    with_stage(ctx, groups * per_group, stage_len * static_cast<int64_t>(sizeof(T)),
               [&](int64_t s, char* stage_raw) {
                   T* const stage = reinterpret_cast<T*>(stage_raw);
                   const int64_t gg = s / per_group;
                   const int64_t rem = s % per_group;
                   const int64_t ob = rem / num_ic_blocks;
                   const int64_t ib = rem % num_ic_blocks;

                   const int64_t icn = ib * g.icn_block;
                   const int64_t icnc = std::min(g.icn_block, ic_per_group - icn);
                   const int64_t kdim = icnc * karea;
                   const int64_t slot =
                       gemm_panel_slot(kdim, g.nr, sizeof(T), false, std::is_same_v<T, bfloat16>);
                   const int64_t oc0 = ob * g.ocn_block;
                   const int64_t oc1 = std::min(oc0 + g.ocn_block, oc_per_group);

                   // HWIO: one input channel's tap is one OC-length run; the group's output
                   // channels start at gg·oc_per_group.
                   const int64_t w_ic = weight.shape(3);      // between input channels
                   const int64_t w_tap = ic_per_group * w_ic; // between filter taps
                   for (int64_t ocn = oc0; ocn < oc1; ocn += g.nr) {
                       const int64_t ocnc = std::min(g.nr, oc_per_group - ocn);
                       impl::pack_block<T>(blob + g.offset(gg, ib, ocn), slot, stage,
                                                   w + gg * oc_per_group + ocn + icn * w_ic, w_ic,
                                                   w_tap, ocnc, icnc, static_cast<int>(karea), g.nr,
                                                   static_cast<int>(kdim));
                   }
               });
}

/// pack_block_i8's sibling of the float pack above: same split, same loop.
void pack_weight_i8(const TensorView& weight, TensorView& packed,
                         const ConvPackedWeight& g, int64_t ic_per_group, int64_t oc_per_group,
                         int64_t groups, const ComputeContext& ctx)
{
    const int64_t karea = g.kdim_max / g.icn_block;
    const int64_t num_ic_blocks = g.nfull_ic + (g.tail_ic ? 1 : 0);
    const int64_t num_oc_blocks = (oc_per_group + g.ocn_block - 1) / g.ocn_block;
    const int64_t per_group = num_oc_blocks * num_ic_blocks;
    const bool a_biased = arch::mma_i8_folds_a_bias;

    const int8_t* const w = weight.ptr<int8_t>();
    int8_t* const blob = packed.ptr<int8_t>();
    const int64_t stage_len = g.nr * g.kdim_max; // bytes

    with_stage(ctx, groups * per_group, stage_len, [&](int64_t s, char* stage_raw) {
        int8_t* const stage = reinterpret_cast<int8_t*>(stage_raw);
        const int64_t gg = s / per_group;
        const int64_t rem = s % per_group;
        const int64_t ob = rem / num_ic_blocks;
        const int64_t ib = rem % num_ic_blocks;

        const int64_t icn = ib * g.icn_block;
        const int64_t icnc = std::min(g.icn_block, ic_per_group - icn);
        const int64_t kdim = icnc * karea;
        const int64_t slot = gemm_panel_slot(kdim, g.nr, 1, true);
        const int64_t oc0 = ob * g.ocn_block;
        const int64_t oc1 = std::min(oc0 + g.ocn_block, oc_per_group);

        const int64_t w_ic = weight.shape(3);
        const int64_t w_tap = ic_per_group * w_ic;
        for (int64_t ocn = oc0; ocn < oc1; ocn += g.nr) {
            const int64_t ocnc = std::min(g.nr, oc_per_group - ocn);
            impl::pack_block_i8(blob + g.offset(gg, ib, ocn), slot, stage,
                                        w + gg * oc_per_group + ocn + icn * w_ic, w_ic, w_tap, ocnc,
                                        icnc, static_cast<int>(karea), g.nr, static_cast<int>(kdim),
                                        a_biased);
        }
    });
}

#if defined(NNOPS_ARCH_SME)
/// The ZA blob (PackedHWIOSme): one whole-K 64-column block per (group, ic
/// block, oc block), so compute() reads it with no per-call pack. The block
/// stride is the kdim_max slot; the last k block packs shorter but stays in it.
template <class T>
void pack_weight_sme(const TensorView& weight, TensorView& packed, const ConvPackedWeight& g,
                     int64_t ic_per_group, int64_t oc_per_group, int64_t groups,
                     const ComputeContext& ctx)
{
    (void)ctx;
    const int64_t karea = g.kdim_max / g.icn_block;
    const int64_t OC = weight.shape(3);
    const int64_t w_ic = OC;
    const int64_t w_tap = ic_per_group * w_ic;
    const T* const w = weight.ptr<T>();
    T* const blob = packed.ptr<T>();
    const int64_t nr = g.nr;
    PoolArray<T> stage(karea > 1 ? static_cast<size_t>(g.kdim_max * nr) : 0);

    for (int64_t gg = 0; gg < groups; ++gg) {
        const T* const w_group = w + gg * oc_per_group;
        for (int64_t ib = 0; ib < g.num_ic_blocks; ++ib) {
            const int64_t icn = ib * g.icn_block;
            const int64_t icnc = std::min(g.icn_block, ic_per_group - icn);
            const int64_t kdim = icnc * karea;
            for (int64_t ob = 0; ob < g.num_oc_blocks; ++ob) {
                const int64_t ocn = ob * nr;
                const int64_t np = std::min(nr, oc_per_group - ocn);
                T* const out = blob +
                               ((gg * g.num_ic_blocks + ib) * g.num_oc_blocks + ob) *
                                   g.sme_region;
                const T* const bsrc = w_group + ocn + icn * w_ic;
                if (karea == 1) {
                    if constexpr (std::is_same_v<T, int8_t>) {
                        tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out, bsrc,
                                             static_cast<int>(w_ic));
                    } else {
                        tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out, bsrc,
                                          static_cast<int>(w_ic));
                    }
                } else {
                    for (int t = 0; t < karea; ++t) {
                        for (int64_t i = 0; i < icnc; ++i) {
                            std::memcpy(stage.data() +
                                            (static_cast<int64_t>(t) * icnc + i) * nr,
                                        bsrc + t * w_tap + i * w_ic,
                                        static_cast<size_t>(np) * sizeof(T));
                        }
                    }
                    if constexpr (std::is_same_v<T, int8_t>) {
                        tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out,
                                             stage.data(), static_cast<int>(nr));
                    } else {
                        tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out, stage.data(),
                                          static_cast<int>(nr));
                    }
                }
            }
        }
    }
}
#endif

/// Σ_k weight[oc, k] per output channel over a group's whole K — the table the
/// int8 epilogue compensates with. Written from the *raw* HWIO filter.
void write_weight_sums(const int8_t* w, int64_t OC, int64_t ic_per_group, int64_t karea,
                       const TensorView& weight, int32_t* out, const ComputeContext& ctx)
{
    // [KH][KW][IC/G][OC]: the walk covers the group's input channels and
    // *all* groups' output channels at once — the OC axis spans groups.
    const int64_t w_ic = weight.shape(3);
    const int64_t w_tap = ic_per_group * w_ic;
    // One output channel is one task: a channel's taps are strided, but the
    // writes are disjoint and this is the pack's smallest cost.
    ctx.cpu.run(0, OC, [&](int64_t oc) {
        int32_t acc = 0;
        for (int64_t t = 0; t < karea; ++t) {
            const int8_t* const row = w + t * w_tap + oc;
            for (int64_t i = 0; i < ic_per_group; ++i) {
                acc += row[i * w_ic];
            }
        }
        out[oc] = acc;
    });
}

} // anonymous namespace

void conv_packed_weight_pack(const Conv2DAttributes& attrs, const TensorView& weight,
                             TensorView& packed, const ComputeContext& ctx)
{
    const int64_t OC = weight.shape(3);
    const int64_t ICG = weight.shape(2);
    const int64_t G = attrs.groups;
    const int64_t OCG = OC / G;
    const DataType dtype = weight.data_type();
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];

    if (packed.layout() == TensorLayout::PackedHWIOSme) {
#if defined(NNOPS_ARCH_SME)
        const ConvPackedWeight g = conv_packed_weight_sme(attrs, ICG, OCG, dtype);
        NNOPS_ASSERT(g.sme && packed.numel() >= g.total);
        if (dtype == DataType::f32) {
            pack_weight_sme<float>(weight, packed, g, ICG, OCG, G, ctx);
        } else if (dtype == DataType::f16) {
            pack_weight_sme<half>(weight, packed, g, ICG, OCG, G, ctx);
        } else {
            NNOPS_ASSERT(dtype == DataType::s8);
            pack_weight_sme<int8_t>(weight, packed, g, ICG, OCG, G, ctx);
        }
        return;
#else
        NNOPS_ASSERT(!"conv_packed_weight_pack: SME blob on a non-SME build");
        return;
#endif
    }

    // No zero-fill here: the panel packers mask-load a short panel and zero-fill
    // the lanes past its width, so every byte a consumer reads is written.
    const ConvPackedWeight g = conv_packed_weight(attrs, ICG, OCG, dtype);
    NNOPS_ASSERT(packed.numel() >= g.total &&
                 "conv_packed_weight_pack: buffer smaller than the panels");
    switch (dtype) {
    case DataType::f32:
        pack_weight<float>(weight, packed, g, ICG, OCG, G, ctx);
        break;
    case DataType::f16:
        pack_weight<half>(weight, packed, g, ICG, OCG, G, ctx);
        break;
#ifdef NNOPS_ARCH_AARCH64
    case DataType::bf16:
        pack_weight<bfloat16>(weight, packed, g, ICG, OCG, G, ctx);
        break;
#endif
    case DataType::s8:
        pack_weight_i8(weight, packed, g, ICG, OCG, G, ctx);
        break;
    default:
        NNOPS_ASSERT(!"conv_packed_weight_pack: unsupported filter dtype");
        return;
    }
    if (is_quantized_conv2d(dtype)) {
        // The sums land on an unaligned byte offset: accumulate whole, copy in.
        PoolArray<int32_t> sums;
        sums.zero(static_cast<size_t>(OC));
        write_weight_sums(weight.ptr<int8_t>(), OC, ICG, karea, weight, sums.data(), ctx);
        std::memcpy(packed.ptr<char>() + g.sums_off, sums.data(),
                    static_cast<size_t>(OC) * sizeof(int32_t));
    }
}

Conv2DGemmPlan get_conv2d_implicit_gemm_plan(const Conv2DAttributes& attrs,
                                             std::span<const TensorDesc> inputs,
                                             std::span<const TensorDesc> outputs, int num_threads,
                                             bool use_thread_slots, int64_t num_slots)
{
    return GemmRoute::make_plan(attrs, inputs, outputs, num_threads, use_thread_slots,
                                       num_slots);
}

Status conv2d_implicit_gemm_kernel(const Conv2DAttributes& attrs, TensorView& output,
                                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                                   void* /*workspace*/)
{
    return conv2d_gemm_run<GemmRoute>(attrs, output, inputs, ctx);
}

} // namespace nnops::backend::cpu
