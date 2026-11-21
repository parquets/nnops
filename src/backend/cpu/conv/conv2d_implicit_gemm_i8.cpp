/// @file conv2d_implicit_gemm_i8.cpp
/// @brief Conv2D as implicit GEMM over NHWC activations — the quantized route.
///
/// The same (pixels on M, packed oc on N, k tap-major) product as the float
/// route, with integer operands: A is the packed activation tile (bytes, the A
/// shift already applied), B the packed s8 HWIO weight, C an int32 accumulator
/// (one read-modify-write pass per ic block over the task's oc slice). The
/// epilogue withdraws both zero points and then dequantizes into fp or
/// requantizes to s8/u8. An fp activation is materialized into a compact s8
/// plane once per call; a pre-quantized one is shifted on the way into the
/// gather instead.
///
/// This file owns the integer tiles end to end: the activation plane, the s8
/// panel packer and the per-call SME prepack, the task body, and the s8
/// filter's load-time prepack. The float route and everything the two families
/// share — the plan, the scratch layout, the shared driver, the prepacked
/// filter's geometry — live in conv2d_implicit_gemm.cpp.
///
/// What not to "fix" later:
///
///  * The MMA is handed a *packed* A panel: direct-A costs ~2.7x the uops, so
///    the gather writes into the panel (or into the row-major tile the same
///    slot's packer reads) rather than feeding the kernel rows in place.
///  * The activation plane exists so the GEMM sees one zero-point convention:
///    an fp activation is quantized into it, a u8 one shifted to s8 with its
///    zero point moved by -128, and a dense s8 one read in place.
///  * R_b is read from the prepacked blob when there is one; the serial walk
///    over the raw filter is the fallback, because a second pool dispatch
///    costs more than the pass (1.18x at 2T, 1.9x at 20T).
///  * On AArch64 SME the SMOPA tile is 16 rows x 64 columns, so a task's A is
///    packed into [m/16][k/4][256] and B is prepacked per (group, task oc
///    start, ic block) — the task starts need not be 64-aligned.

#include "conv2d_implicit_gemm_internal.h"
#include "backend/cpu/matmul/matmul_helper.h" // uniform pack/MMA + panel strides
#if defined(NNOPS_ARCH_SME)
#include "backend/cpu/aarch64/sme/mma_sme.hpp" // available_sme + SME tile geometry
#endif
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"    // dequant / requant epilogues
#include "backend/cpu/simd_kernel/simd_im2col.hpp"      // tiled im2col gathers
#include "backend/cpu/simd_kernel/simd_int8_reduce.hpp" // int8_row_sum / int8_col_sum_slab
#include "backend/cpu/simd_kernel/simd_quant.hpp"       // quant_kernel::quantization
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace nnops::backend::cpu {

namespace impl {
namespace {

using nnops::simd::s_load;
using nnops::simd::s_store;

#if defined(NNOPS_ARCH_SME)
/// The quantized twin of the float gather: a ZA row is one pixel's channel run (four channels a
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
#endif  // NNOPS_ARCH_SME

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

/// conv2d_implicit_gemm.cpp's pack_block, over int8: same panel, same tap-major
/// restaging.
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
    // The SMOPA blob's n block is the ZA tile width (64), not the NEON panel's
    // NR_MAX_I8; only a panel blob has to match the micro-kernel's nr.
    NNOPS_ASSERT(c.pg.icn_block == c.icn_block && c.pg.ocn_block == c.ocn_block &&
                 (c.sme_blob || c.pg.nr == NR_MAX_I8));
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

} // anonymous namespace

/// The quantized ladder: the output dtype, then the activation's.
Status conv2d_quant_impl(const Conv2DAttributes& attrs, TensorView& output,
                         std::span<const TensorView> inputs, const ComputeContext& ctx,
                         const Conv2DGemmPlan& plan, void* workspace)
{
    const auto out = output.data_type();
    switch (inputs[0].data_type()) {
    case DataType::f32:
        switch (out) {
        case DataType::f32:
            conv2d_int8_impl<float, float>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::f16:
            conv2d_int8_impl<float, half>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::s8:
            conv2d_int8_impl<float, int8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::u8:
            conv2d_int8_impl<float, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        default:
            return Status::unsupported;
        }
    case DataType::f16:
        switch (out) {
        case DataType::f32:
            conv2d_int8_impl<half, float>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::f16:
            conv2d_int8_impl<half, half>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::s8:
            conv2d_int8_impl<half, int8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::u8:
            conv2d_int8_impl<half, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        default:
            return Status::unsupported;
        }
    case DataType::u8:
        switch (out) {
        case DataType::f32:
            conv2d_int8_impl<uint8_t, float>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::f16:
            conv2d_int8_impl<uint8_t, half>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::s8:
            conv2d_int8_impl<uint8_t, int8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::u8:
            conv2d_int8_impl<uint8_t, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        default:
            return Status::unsupported;
        }
    case DataType::s8:
        switch (out) {
        case DataType::f32:
            conv2d_int8_impl<int8_t, float>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::f16:
            conv2d_int8_impl<int8_t, half>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::s8:
            conv2d_int8_impl<int8_t, int8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        case DataType::u8:
            conv2d_int8_impl<int8_t, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
            return Status::ok;
        default:
            return Status::unsupported;
        }
    default:
        return Status::unsupported;
    }
}

} // namespace impl

// =========================================================================
//  The prepacked filter — the s8 panels and the R_b table
// =========================================================================
//
// The float panels' prepack lives with the float route; this is the int8 half:
// the same (group, oc block, k block) split, the int8 panel packer, and the
// per-output-channel sum table the epilogue compensates with.

namespace {

/// The float prepack's sibling (conv2d_implicit_gemm.cpp): same split, same loop.
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

    impl::with_stage(ctx, groups * per_group, stage_len, [&](int64_t s, char* stage_raw) {
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

#if defined(NNOPS_ARCH_SME)
/// The ZA blob's int8 panels (PackedHWIOSme): one whole-K 64-column block per
/// (group, ic block, oc block), so compute() reads it with no per-call pack.
/// The block stride is the kdim_max slot; the last k block packs shorter but
/// stays in it.
void pack_weight_sme_i8(const TensorView& weight, TensorView& packed, const ConvPackedWeight& g,
                        int64_t ic_per_group, int64_t oc_per_group, int64_t groups)
{
    const int64_t karea = g.kdim_max / g.icn_block;
    const int64_t OC = weight.shape(3);
    const int64_t w_ic = OC;
    const int64_t w_tap = ic_per_group * w_ic;
    const int8_t* const w = weight.ptr<int8_t>();
    int8_t* const blob = packed.ptr<int8_t>();
    const int64_t nr = g.nr;
    PoolArray<int8_t> stage(karea > 1 ? static_cast<size_t>(g.kdim_max * nr) : 0);

    for (int64_t gg = 0; gg < groups; ++gg) {
        const int8_t* const w_group = w + gg * oc_per_group;
        for (int64_t ib = 0; ib < g.num_ic_blocks; ++ib) {
            const int64_t icn = ib * g.icn_block;
            const int64_t icnc = std::min(g.icn_block, ic_per_group - icn);
            const int64_t kdim = icnc * karea;
            for (int64_t ob = 0; ob < g.num_oc_blocks; ++ob) {
                const int64_t ocn = ob * nr;
                const int64_t np = std::min(nr, oc_per_group - ocn);
                int8_t* const out = blob +
                                    ((gg * g.num_ic_blocks + ib) * g.num_oc_blocks + ob) *
                                        g.sme_region;
                const int8_t* const bsrc = w_group + ocn + icn * w_ic;
                if (karea == 1) {
                    tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out, bsrc,
                                         static_cast<int>(w_ic));
                } else {
                    for (int t = 0; t < karea; ++t) {
                        for (int64_t i = 0; i < icnc; ++i) {
                            std::memcpy(stage.data() +
                                            (static_cast<int64_t>(t) * icnc + i) * nr,
                                        bsrc + t * w_tap + i * w_ic,
                                        static_cast<size_t>(np));
                        }
                    }
                    tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out, stage.data(),
                                         static_cast<int>(nr));
                }
            }
        }
    }
}
#endif

} // anonymous namespace

void conv_packed_weight_pack_i8(const TensorView& weight, TensorView& packed,
                                const ConvPackedWeight& g, int64_t ic_per_group,
                                int64_t oc_per_group, int64_t groups, const ComputeContext& ctx)
{
    if (packed.layout() == TensorLayout::PackedHWIOSme) {
#if defined(NNOPS_ARCH_SME)
        pack_weight_sme_i8(weight, packed, g, ic_per_group, oc_per_group, groups);
        return;
#else
        NNOPS_ASSERT(!"conv_packed_weight_pack_i8: SME blob on a non-SME build");
        return;
#endif
    }

    // No zero-fill here: the panel packers mask-load a short panel and zero-fill
    // the lanes past its width, so every byte a consumer reads is written.
    NNOPS_ASSERT(packed.numel() >= g.total &&
                 "conv_packed_weight_pack_i8: buffer smaller than the panels");
    pack_weight_i8(weight, packed, g, ic_per_group, oc_per_group, groups, ctx);

    // The R_b table trails the panels at an unaligned byte offset, so it is
    // accumulated whole and copied in.
    const int64_t OC = weight.shape(3);
    const int64_t karea = g.kdim_max / g.icn_block;
    PoolArray<int32_t> sums;
    sums.zero(static_cast<size_t>(OC));
    write_weight_sums(weight.ptr<int8_t>(), OC, ic_per_group, karea, weight, sums.data(), ctx);
    std::memcpy(packed.ptr<char>() + g.sums_off, sums.data(),
                static_cast<size_t>(OC) * sizeof(int32_t));
}

} // namespace nnops::backend::cpu
