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

// ---- main kernel ----------------------------------------------------------

template <class T>
void conv2d_impl(const Conv2DAttributes& attrs, TensorView& output,
                              std::span<const TensorView> inputs, const ComputeContext& ctx,
                              const Conv2DGemmPlan& plan, void* workspace)
{
    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Logical [N, C, H, W]; NHWC stores [N][H][W][C], so a "row" is one spatial
    // line and a pixel's channel vector is contiguous.
    const int64_t N = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // HWIO: the output channel is the last axis; a prepacked blob leads with it.
    const bool sme_blob = weight.layout() == TensorLayout::PackedHWIOSme;
    const bool prepacked = is_packed_conv_weight(weight.layout());
    const int64_t OC = prepacked ? weight.shape(0) : weight.shape(3);

    // The ZA tiles read raw HWIO filters in their own layout, and their own
    // prepacked form directly. A PackedHWIO blob is the NEON panels and has no
    // reader here, so that case stays on the panel route.
#if defined(NNOPS_ARCH_SME)
    constexpr bool sme_dtype = std::is_same_v<T, float> || std::is_same_v<T, half>;
    constexpr DataType sme_be_dt = std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
#endif

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

#if defined(NNOPS_ARCH_SME)
    // Only wide-OC blocks fill four 64-column ZA tiles; below that the per-tile
    // A pack and the half-empty B blocks lose to the NEON panels. A caller's
    // prepacked HWIO blob is the NEON layout and stays there too.
    // 128 output channels fill four 32-column tiles: enough for a pointwise
    // filter, whose A unpack is shared across every oc block. A k-tap filter
    // repacks A per (oc block, k block), so it needs the wider 64-column gate
    // — and a narrow row's partial last tile wastes the 32-row kernel.
    const int64_t sme_oc_min = aarch64::sme_fp_nr;
    const bool use_sme =
        sme_dtype && (sme_blob || (!prepacked && aarch64::available_sme() &&
                                   oc_per_group >= sme_oc_min));
#else
    constexpr bool use_sme = false;
#endif

    const int KH = static_cast<int>(attrs.kernel_size[0]);
    const int KW = static_cast<int>(attrs.kernel_size[1]);
    const int SH = static_cast<int>(attrs.stride[0]);
    const int SW = static_cast<int>(attrs.stride[1]);
    const int DH = static_cast<int>(attrs.dilation[0]);
    const int DW = static_cast<int>(attrs.dilation[1]);
    const int PH = static_cast<int>(attrs.padding[0]);
    const int PW = static_cast<int>(attrs.padding[1]);
    const int karea = KH * KW;

    const int64_t icn_block = plan.icn_block;
    const int64_t ocn_block = plan.ocn_block;
    const int m_block = static_cast<int>(plan.m_block);
    const int64_t oh_chunks = plan.oh_chunks;
    const int64_t ph_block = plan.ph_block;
    const int64_t oc_chunks = plan.oc_chunks;

    // A sample is IH rows of the NHWC pitch.
    const int64_t ib_step = IH * in_row_stride;
    const int64_t ob_step = OH * out_row_stride;

    // HWIO (TensorFlow's filter): [KH][KW][IC/G][OC]. The output channel is the
    // contiguous axis, so one (tap, input channel) is an OC-length run and the B
    // panel is a plain strided copy — no transpose. Dims, not pitch: a weight is
    // a load-time constant. A prepacked blob is read only through the panel
    // geometry, so its strides stay 0.
    //   w_ic_stride = OC (one input channel's row); w_tap_stride = ic_per_group·OC
    const int64_t w_row = prepacked ? 0 : weight.shape(3); // [oc] run of one (tap, ic)
    const int64_t w_ic_stride = w_row;                     // [ic] within a tap
    const int64_t w_tap_stride = prepacked                 // [kh][kw] position
                                     ?
                                     0 :
                                     weight.shape(2) * w_row;

    T* out_ptr = output.ptr<T>();
    const T* in_ptr = input.ptr<T>();
    const T* w_ptr = weight.ptr<T>();
    const T* b_ptr = has_bias ? inputs[2].ptr<T>() : nullptr;

    constexpr int64_t nr_max = nr_max_flt<T>();
    const int64_t esz = static_cast<int64_t>(sizeof(T));
    constexpr float kInf = std::numeric_limits<float>::infinity();

    // ---- the finish decision ----------------------------------------------
    // A pure clamp folds into the MMA store. The bias is per *column* here
    // (rows are pixels), so it seeds the tile instead of the micro-kernel's
    // per-row seed — leaving the clamp nothing else to do.
    // A pre-activation residual joins the sum before the clamp, so the clamp
    // cannot ride the MMA store and the epilogue stays a separate pass.
    const bool pre_act = attrs.add_to && attrs.residual_pre_activation;

    float clamp_lo = -kInf;
    float clamp_hi = kInf;
    bool epi_folded = false;
    if (!attrs.add_to || pre_act) {
        switch (attrs.epilogue.type) {
        case EpilogueActivateType::Relu:
            clamp_lo = 0.0f;
            epi_folded = true;
            break;
        case EpilogueActivateType::Relu6:
            clamp_lo = 0.0f;
            clamp_hi = 6.0f;
            epi_folded = true;
            break;
        default:
            break;
        }
    }
    const bool epi_active = attrs.epilogue.type != EpilogueActivateType::None;
    const bool epi_pass = epi_active && !epi_folded;

    // add_to folds the pre-existing output into the accumulator seed when the
    // epilogue is the identity, and the pre-activation order always does; the
    // post-activation order with an activation preserves the original tile and
    // adds it after the epilogue instead.
    const bool keep_orig = attrs.add_to && epi_pass && !pre_act;

    // The bias is per column, so it always seeds the tile; add_to's seed also
    // carries the residual. A seeded tile must never be zeroed by the MMA.
    const bool seed_bias = attrs.add_to && has_bias;
    const bool bias_seed = has_bias && !attrs.add_to;
    const bool need_epi = epi_pass;

    // ---- the tiny filter ---------------------------------------------------
    // At most four input channels under a square 3x3/5x5/7x7: no gather, no
    // panel, no plan — the tap is read where it lies. prepackWeights declines
    // this shape too, so the filter arrives raw.
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
        if (G == 1 && !prepacked && ic_per_group <= 4 && oc_per_group % 16 == 0 && KH == KW &&
            KH >= 3 && KH <= 7 && (KH & 1) && !attrs.add_to && !epi_pass) {
            const bool clamp = clamp_lo != -kInf || clamp_hi != kInf;
            const float lo = clamp ? clamp_lo : -kInf;
            const float hi = clamp ? clamp_hi : kInf;
            const auto run = [&](auto kk) {
                constexpr int K = decltype(kk)::value;
                switch (ic_per_group) {
                case 1:
                    stem_conv_plane<T, 1, K>(out_ptr, in_ptr, w_ptr, b_ptr, N, IH, IW, OW, OC,
                                             in_row_stride, out_row_stride, ib_step, ob_step, OH,
                                             SH, SW, PH, PW, DH, DW, clamp, lo, hi, ctx);
                    return;
                case 2:
                    stem_conv_plane<T, 2, K>(out_ptr, in_ptr, w_ptr, b_ptr, N, IH, IW, OW, OC,
                                             in_row_stride, out_row_stride, ib_step, ob_step, OH,
                                             SH, SW, PH, PW, DH, DW, clamp, lo, hi, ctx);
                    return;
                case 3:
                    stem_conv_plane<T, 3, K>(out_ptr, in_ptr, w_ptr, b_ptr, N, IH, IW, OW, OC,
                                             in_row_stride, out_row_stride, ib_step, ob_step, OH,
                                             SH, SW, PH, PW, DH, DW, clamp, lo, hi, ctx);
                    return;
                default:
                    stem_conv_plane<T, 4, K>(out_ptr, in_ptr, w_ptr, b_ptr, N, IH, IW, OW, OC,
                                             in_row_stride, out_row_stride, ib_step, ob_step, OH,
                                             SH, SW, PH, PW, DH, DW, clamp, lo, hi, ctx);
                    return;
                }
            };
            switch (KH) {
            case 3:
                run(std::integral_constant<int, 3>{});
                return;
            case 5:
                run(std::integral_constant<int, 5>{});
                return;
            default:
                run(std::integral_constant<int, 7>{});
                return;
            }
        }
    }

    // ---- block geometry ----------------------------------------------------
    // Same function prepackWeights built the blob with; it must never disagree.
    const ConvPackedWeight pg = sme_blob ?
                                    conv_packed_weight_sme(attrs, ic_per_group, oc_per_group,
                                                           weight.data_type()) :
                                    conv_packed_weight(attrs, ic_per_group, oc_per_group,
                                                       weight.data_type());
    if (!sme_blob) {
        NNOPS_ASSERT(pg.icn_block == icn_block && pg.ocn_block == ocn_block && pg.nr == nr_max);
    }
    const int64_t num_ic_blocks = pg.nfull_ic + (pg.tail_ic ? 1 : 0);
    const int64_t kdim_max = pg.kdim_max;

    // ---- scratch -----------------------------------------------------------
    const Conv2DGemmScratch sc = gemm_scratch_layout(plan, esz);
    char* const base = static_cast<char*>(workspace);
    T* const shared_ptr = reinterpret_cast<T*>(base + sc.panel_off);
    const bool shared = plan.shared_elems > 0;

    // The panels every task reads in place: the caller's blob, or this call's
    // prepack in the arena (null when the weight did not fit — each task then
    // packs its own into its slot).
    const T* const panels =
        (!use_sme && (prepacked || shared)) ? (prepacked ? w_ptr : shared_ptr) : nullptr;

    // ---- the shared weight prepack (once per call) -------------------------
    // The same entry prepackWeights uses, only the destination differs. The SME
    // route packs its own B layout per task, so the NEON prepack is skipped.
    if (shared && !use_sme) {
        const int64_t cols = (plan.shared_elems + OC - 1) / OC;
        const int64_t shape[] = {OC, cols};
        TensorView blob(std::span<const int64_t>(shape, 2), weight.data_type(), shared_ptr,
                        cols * esz, TensorLayout::PackedHWIO);
        conv_packed_weight_pack(attrs, weight, blob, ctx);
    }

    const int64_t NG = N * G;
    const int64_t chunks_per_ng = oh_chunks * oc_chunks;
    const int64_t total = NG * chunks_per_ng;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < total;

    // The micro-kernel loads a bias panel at a time, so a group's bias must be
    // readable `nr_max` past its last channel. One padded copy covers every
    // group: lanes past a group are only read, and the store drops them.
    PoolArray<T> bias_pad;
    if (bias_seed) {
        bias_pad.assign(static_cast<size_t>(OC + nr_max), T{});
        std::memcpy(bias_pad.data(), b_ptr, static_cast<size_t>(OC) * sizeof(T));
    }

#if defined(NNOPS_ARCH_SME)
    // ---- the shared SME filter prepack (once per call) ----------------------
    // One SME-layout copy per (group, task oc start, ic block), so the task
    // grid reads it in place instead of re-packing its slice per oh chunk. The
    // start list is the tasks' own oc tiling — `ocn_block` need not be a
    // 64-column multiple, so a per-chunk layout could not be addressed. Blocks
    // inside a region are `sme_slot_max` apart and the kernel is told that
    // stride.
    std::vector<int64_t> sme_start_prefix;
    PoolArray<T> sme_b_shared;
    PoolArray<T> sme_stage_shared;
    int64_t sme_total_starts = 0;
    int64_t sme_region_elems = 0;
    int64_t sme_slot_max = 0;
    int64_t sme_num_ic_blocks = 0;
    if constexpr (sme_dtype) {
        if (use_sme && !sme_blob) {
            const int64_t nr = aarch64::sme_fp_nr;
            sme_num_ic_blocks = (ic_per_group + icn_block - 1) / icn_block;
            sme_start_prefix.assign(static_cast<size_t>(oc_chunks) + 1, 0);
            for (int64_t c = 0; c < oc_chunks; ++c) {
                const int64_t lo = c * pg.panels_oc / oc_chunks * nr_max;
                const int64_t hi =
                    std::min((c + 1) * pg.panels_oc / oc_chunks * nr_max, oc_per_group);
                const int64_t starts = (hi - lo + ocn_block - 1) / ocn_block;
                sme_start_prefix[static_cast<size_t>(c) + 1] =
                    sme_start_prefix[static_cast<size_t>(c)] + starts;
            }
            sme_total_starts = sme_start_prefix[static_cast<size_t>(oc_chunks)];
            sme_slot_max = sme_b_panel_stride(sme_be_dt, kdim_max);
            const int64_t max_blocks = (ocn_block + nr - 1) / nr;
            sme_region_elems = max_blocks * sme_slot_max;
            const int64_t slices = G * sme_total_starts * sme_num_ic_blocks;
            sme_b_shared.reset(static_cast<size_t>(slices * sme_region_elems));
            if (karea > 1) {
                sme_stage_shared.reset(static_cast<size_t>(kdim_max * nr));
            }
            for (int64_t g = 0; g < G; ++g) {
                const T* const w_group = w_ptr + g * oc_per_group;
                for (int64_t c = 0; c < oc_chunks; ++c) {
                    const int64_t oc_lo = c * pg.panels_oc / oc_chunks * nr_max;
                    const int64_t oc_hi =
                        std::min((c + 1) * pg.panels_oc / oc_chunks * nr_max, oc_per_group);
                    int64_t si = 0;
                    for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block, ++si) {
                        const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);
                        for (int64_t ib = 0; ib < sme_num_ic_blocks; ++ib) {
                            const int64_t icn0 = ib * icn_block;
                            const int64_t icnc = std::min(icn_block, ic_per_group - icn0);
                            const int64_t kdim = icnc * karea;
                            T* const dst =
                                sme_b_shared.data() +
                                ((g * sme_total_starts + sme_start_prefix[static_cast<size_t>(c)] +
                                  si) *
                                     sme_num_ic_blocks +
                                 ib) *
                                    sme_region_elems;
                            for (int64_t p0 = 0; p0 < ocnc; p0 += nr) {
                                const int64_t np = std::min(nr, ocnc - p0);
                                const T* const bsrc =
                                    w_group + ocn + p0 + icn0 * w_ic_stride;
                                T* const out = dst + (p0 / nr) * sme_slot_max;
                                if (karea == 1) {
                                    tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out,
                                                      bsrc, static_cast<int>(w_ic_stride));
                                } else {
                                    for (int t = 0; t < karea; ++t) {
                                        for (int64_t i = 0; i < icnc; ++i) {
                                            std::memcpy(
                                                sme_stage_shared.data() +
                                                    (static_cast<int64_t>(t) * icnc + i) * nr,
                                                bsrc + t * w_tap_stride + i * w_ic_stride,
                                                static_cast<size_t>(np) * sizeof(T));
                                        }
                                    }
                                    tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out,
                                                      sme_stage_shared.data(),
                                                      static_cast<int>(nr));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    const auto run_task = [&](int64_t t) {
        const int64_t ng = t / chunks_per_ng;
        const int64_t rem = t % chunks_per_ng;
        // The oc-chunk varies fastest: neighbours in task order then cover
        // different filter rows, so their weight streams differ.
        const int64_t chunk = rem / oc_chunks;
        const int64_t oc_chunk = rem % oc_chunks;
        const int64_t n = ng / G;
        const int64_t g = ng % G;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;
        char* const slot_ptr = base + sc.head + slot * sc.slot_stride;
        T* const a_buf = reinterpret_cast<T*>(slot_ptr + sc.off_a);
        T* const orig_buf = reinterpret_cast<T*>(slot_ptr + sc.off_orig);
        T* const stage = reinterpret_cast<T*>(slot_ptr + sc.off_stage);
        T* const pack_buf = reinterpret_cast<T*>(slot_ptr + sc.off_pack);

        // The SME route's per-task A slab; B is the shared prepack above.
#if defined(NNOPS_ARCH_SME)
        PoolArray<T> sme_a;
        if constexpr (sme_dtype) {
            if (use_sme) {
                constexpr int64_t kSmeARows = 2 * aarch64::sme_mr;
                const int64_t a_blocks = (m_block + kSmeARows - 1) / kSmeARows;
                sme_a.reset(static_cast<size_t>(
                    a_blocks * sme_a_panel_stride(sme_be_dt, kdim_max)));
            }
        }
#endif

        T* const out_g = out_ptr + n * ob_step + g * oc_per_group;
        const T* const in_g = in_ptr + n * ib_step + g * ic_per_group;
        const T* const w_g = w_ptr + g * oc_per_group;
        const T* const bias_g = b_ptr ? b_ptr + g * oc_per_group : nullptr;
        const T* const bias_col = bias_seed ? bias_pad.data() + g * oc_per_group : nullptr;

        // This oh-chunk's contiguous row range, and its oc range. The oc split
        // is over *panels*, not channels: the blob holds panel p as channels
        // [p·nr, (p+1)·nr) of its k column, so a task starts where a panel does.
        const int64_t r0 = chunk * OH / oh_chunks;
        const int64_t r1 = (chunk + 1) * OH / oh_chunks;
        // An SME blob keeps the whole group on the filter axis (its plan sets
        // oc_chunks 1): starts and the 64-column blocks then agree.
        const int64_t oc_lo =
            sme_blob ? 0 : oc_chunk * pg.panels_oc / oc_chunks * nr_max;
        const int64_t oc_hi =
            sme_blob ? oc_per_group :
                       std::min((oc_chunk + 1) * pg.panels_oc / oc_chunks * nr_max,
                                oc_per_group);

        // Dense rows make a pixel tile one contiguous run, so it may cross a row
        // boundary — which keeps a narrow plane's tiles at the full M instead of
        // paying MR-row granularity per row (a 7-pixel row of a 6-row MR block
        // wastes 40%). Pointwise reads A the same way, so the input must be dense.
        const bool flat =
            out_row_stride == OW * OC && (!plan.pointwise || in_row_stride == IW * IC);

        // A task may start mid-block (oc_chunks splits on panels), but the blob
        // holds one oc block at a time: end the tile where the block does, so its
        // panels keep the stride the kernel derives from kdim. The step is
        // therefore the tile's own end — a whole block from a short tile would
        // skip the rest of that block.
        const auto block_end = [&](int64_t ocn) {
            return panels ? std::min(oc_hi, (ocn / ocn_block + 1) * ocn_block) : oc_hi;
        };
        for (int64_t ocn = oc_lo; ocn < oc_hi; ocn = panels ? block_end(ocn) : ocn + ocn_block) {
            const int64_t oc_end = block_end(ocn);
            const int64_t ocnc = std::min(ocn_block, oc_end - ocn);
            // The residual region covers one pass's rows (one oc block live).
            T* const orig = keep_orig ? orig_buf : nullptr;

            for (int64_t pr = r0; pr < r1; pr += ph_block) {
                const int64_t pr1 = std::min(r1, pr + ph_block);
                const int64_t p_first = pr * OW;
                const int64_t p_last = pr1 * OW;
                // A pass of 49 pixels is 24 + 24 + 1 and the last tile pays a whole
                // MR block for one row. Re-split only a *short* tail, and keep the
                // step a multiple of MR: 24 + 24 + 12 costs the same MR blocks as
                // three even 16s, where a step of 23 would pay a fourth block for 5.
                constexpr int64_t mr_step = mr_max_flt<T>();
                const int64_t p_span = std::max<int64_t>(p_last - p_first, 1);
                const int64_t p_tail = p_span % m_block;
                const int64_t p_step =
                    (flat && p_tail != 0 && p_tail < mr_step && m_block >= 2 * mr_step) ?
                        (m_block / mr_step - 1) * mr_step :
                        m_block;

                for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                    const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                    const bool is_last_icn = (icn + icn_block) >= ic_per_group;
                    const int kdim = static_cast<int>(icnc * karea);

                    const T* b_panel;
                    int b_ld = 0;
                    bool b_sme = false;
                    int64_t sme_stride = 0;
#if defined(NNOPS_ARCH_SME)
                    if constexpr (sme_dtype) {
                        if (use_sme) {
                            if (sme_blob) {
                                // The prepacked blob's block for (group, ic
                                // block, 64-column oc block); consecutive oc
                                // blocks are one region apart.
                                const int64_t slice =
                                    (g * pg.num_ic_blocks + icn / icn_block) *
                                    pg.num_oc_blocks + ocn / pg.nr;
                                b_panel = w_ptr + slice * pg.sme_region;
                                sme_stride = pg.sme_region;
                            } else {
                                // The shared prepack's slice for this task's
                                // (group, oc start, ic block); its first block
                                // is the task's own `ocn`.
                                const int64_t oc_lo_chunk =
                                    oc_chunk * pg.panels_oc / oc_chunks * nr_max;
                                const int64_t starts_before =
                                    sme_start_prefix[static_cast<size_t>(oc_chunk)];
                                const int64_t si_local = (ocn - oc_lo_chunk) / ocn_block;
                                const int64_t slice =
                                    (g * sme_total_starts + starts_before + si_local) *
                                        sme_num_ic_blocks +
                                    icn / icn_block;
                                b_panel = sme_b_shared.data() + slice * sme_region_elems;
                                sme_stride = sme_slot_max;
                            }
                            b_sme = true;
                        }
                    }
#endif
                    if (!b_sme) {
                        if (panels) {
                            // The blob's own walk: this oc block's k block, in
                            // place — no pack, however many chunks the oc axis took.
                            b_panel = panels + pg.offset(g, icn / icn_block, ocn);
                        } else if (karea == 1) {
                            // A 1x1 filter is [ic][oc] with oc contiguous — a
                            // panel's k rows already, `w_ic_stride` apart — so read
                            // it in place and let the stage take the partial last
                            // panel only.
                            b_panel = w_g + ocn + icn * w_ic_stride;
                            b_ld = static_cast<int>(w_ic_stride);
                        } else {
                            pack_block<T>(
                                pack_buf,
                                gemm_panel_slot(kdim, nr_max, esz, false,
                                                std::is_same_v<T, bfloat16>),
                                stage, w_g + ocn + icn * w_ic_stride, w_ic_stride, w_tap_stride,
                                ocnc, icnc, karea, nr_max, kdim);
                            b_panel = pack_buf;
                        }
                    }

                    const bool clamp_here = is_last_icn && epi_folded;
                    const float c_lo = clamp_here ? clamp_lo : -kInf;
                    const float c_hi = clamp_here ? clamp_hi : kInf;

                    for (int64_t p = p_first; p < p_last;) {
                        const int64_t oh = p / OW;
                        const int64_t ow0 = p - oh * OW;
                        const int64_t left = flat ? std::min(p_step, p_last - p) : (OW - ow0);
                        const int mc = static_cast<int>(std::min<int64_t>(m_block, left));

                        const T* a;
                        int lda;
                        if (plan.pointwise) {
                            // The pixel is its own patch: A is the input run itself
                            // (contiguous across a row boundary only when flat).
                            a = in_g + oh * in_row_stride + ow0 * IC + icn;
                            lda = static_cast<int>(IC);
                        } else {
                            tiled_im2col<T>(a_buf, kdim, in_g + icn, in_row_stride, IC,
                                                         T{}, p, static_cast<int>(OW), flat, mc,
                                                         static_cast<int>(IH), static_cast<int>(IW),
                                                         KH, KW, SH, SW, DH, DW, PH, PW,
                                                         static_cast<int>(icnc), karea);
                            a = a_buf;
                            lda = kdim;
                        }

                        T* const tile = out_g + oh * out_row_stride + ow0 * OC + ocn;
                        T* const orig_tile = orig ? orig + (p - p_first) * ocn_block : nullptr;

                        if (icn == 0 && attrs.add_to) {
                            if (keep_orig) {
                                tile_save<T>(orig_tile, tile, static_cast<int>(OC), mc,
                                             static_cast<int>(ocnc));
                            }
                            if (seed_bias) {
                                // The residual stays in the accumulator only on the
                                // identity-epilogue route; keep_orig saves it and
                                // the tile is the seed.
                                tile_seed<T>(tile, static_cast<int>(OC), mc, static_cast<int>(ocnc),
                                             bias_g + ocn,
                                             /*add=*/!keep_orig);
                            } else if (keep_orig) {
                                // No bias: the tile holds the residual, which the
                                // epilogue adds back itself.
                                tile_zero<T>(tile, static_cast<int>(OC), mc,
                                             static_cast<int>(ocnc));
                            }
                        }

                        // Bias only: build the accumulator from it (one panel-lane
                        // load per n-block) rather than seed the tile, which
                        // would double the tile's traffic for one addend.
                        // A 1x1 filter reads in place; everything else arrives as
                        // panels, the blob's or this call's.
                        bool mma_sme = false;
#if defined(NNOPS_ARCH_SME)
                        if constexpr (sme_dtype) {
                            if (use_sme) {
                                tile_pack_lhs_sme(false, mc, kdim, sme_a.data(), a, lda);
                                tile_mma_pack_sme(
                                    mc, static_cast<int>(ocnc), kdim, tile,
                                    static_cast<int>(OC), sme_a.data(), b_panel, sme_stride,
                                    c_lo, c_hi,
                                    /*zero_mode=*/icn == 0 && !attrs.add_to && !bias_seed,
                                    (icn == 0 && bias_seed) ? bias_col + ocn : nullptr);
                                mma_sme = true;
                            }
                        }
#endif
                        if (!mma_sme) {
                            tile_mma_direct(
                                mc, static_cast<int>(ocnc), kdim, tile, static_cast<int>(OC), a,
                                lda, b_panel, b_ld != 0 ? b_ld : -1, stage, c_lo, c_hi,
                                /*zero_mode=*/icn == 0 && !attrs.add_to && !bias_seed, nullptr,
                                (icn == 0 && bias_seed) ? bias_col + ocn : nullptr);
                        }

                        if (is_last_icn && need_epi) {
                            epilogue_inplace<T>(mc, static_cast<int>(ocnc), tile,
                                                static_cast<int>(OC), nullptr, attrs.epilogue,
                                                keep_orig ? orig_tile : nullptr, keep_orig);
                        }
                        p += flat ? p_step : mc;
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, total, run_task);
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

    const int64_t IC = in.shape(1);
    const int64_t IH = in.shape(2);
    const int64_t IW = in.shape(3);
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
                for (int64_t j = 0; j < len; ++j) {
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

// ---- main kernel ----------------------------------------------------------

template <class TAct, class TOut>
void conv2d_int8_impl(const Conv2DAttributes& attrs, TensorView& output,
                                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                                   const Conv2DGemmPlan& plan, void* workspace)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    constexpr bool kU8 = std::is_same_v<TAct, uint8_t>;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    // HWIO: the output channel is the last axis; a prepacked blob leads with it.
    const bool prepacked = is_packed_conv_weight(weight.layout());
    const int64_t OC = prepacked ? weight.shape(0) : weight.shape(3);

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

    // The int8 SMOPA route is implemented but stays off: on MobileNetV2 its
    // per-tile scalar A pack and 64-column blocks lose to the NEON panels at
    // every width (about 2x on the 3x3 256/512 layers). The tiled MatMul route
    // still serves the int8 GEMMs the transpose-conv path issues.
    constexpr bool use_sme = false;

    const int KH = static_cast<int>(attrs.kernel_size[0]);
    const int KW = static_cast<int>(attrs.kernel_size[1]);
    const int SH = static_cast<int>(attrs.stride[0]);
    const int SW = static_cast<int>(attrs.stride[1]);
    const int DH = static_cast<int>(attrs.dilation[0]);
    const int DW = static_cast<int>(attrs.dilation[1]);
    const int PH = static_cast<int>(attrs.padding[0]);
    const int PW = static_cast<int>(attrs.padding[1]);
    const int karea = KH * KW;
    const int64_t k_full = ic_per_group * karea; ///< the epilogue's K

    const int64_t icn_block = plan.icn_block;
    const int64_t ocn_block = plan.ocn_block;
    const int m_block = static_cast<int>(plan.m_block);
    const int64_t oh_chunks = plan.oh_chunks;
    const int64_t ph_block = plan.ph_block;
    const int64_t oc_chunks = plan.oc_chunks;

    const int64_t ob_step = OH * out_row_stride;

    // HWIO: dims, not pitch — one (tap, input channel) is one OC-length run.
    const int64_t w_row = prepacked ? 0 : weight.shape(3);
    const int64_t w_ic_stride = w_row;
    const int64_t w_tap_stride = prepacked ? 0 : weight.shape(2) * w_row;

    // Same geometry prepackWeights used (see the float kernel).
    const ConvPackedWeight pg =
        conv_packed_weight(attrs, ic_per_group, oc_per_group, weight.data_type());
    NNOPS_ASSERT(pg.icn_block == icn_block && pg.ocn_block == ocn_block && pg.nr == NR_MAX_I8);
    constexpr int64_t nr_max = NR_MAX_I8;

    // ---- quantization parameters -----------------------------------------
    const QuantParams& qa = input.quant_params();
    const QuantParams& qw = weight.quant_params();

    const float s_x = quant_scale_at(qa, 0);
    const int32_t zp_x = quant_zp_at(qa, 0);
    NNOPS_ASSERT(s_x > 0.0f);

    // `d`: the shift the A bytes carry into the GEMM. Both arches fold out only
    // the +128 *they* add, so the epilogue must undo `d`. Which applies comes from
    // the *quantization target* — the activation's own storage type when it
    // arrives quantized, else `act_quant_dtype`.
    const bool u8a = kFpAct ? (attrs.act_quant_dtype == DataType::u8) : kU8;
    const int32_t d = u8a ? -128 : 0;

    // The column term exists when the A-side shift and zero point do not cancel;
    // the per-pixel term when some weight zero point is non-zero.
    const bool col_term = (d + zp_x) != 0;
    const bool comp = any_zp_nonzero(qw, OC);

    // ---- the activation as s8 bytes ---------------------------------------
    const Conv2DGemmScratch sc = gemm_scratch_layout(plan, 1);
    char* const base = static_cast<char*>(workspace);

    const bool plane = plan.quant_size > 0;
    uint8_t xr = u8a ? uint8_t{0x80} : uint8_t{0};

    const uint8_t* in_base = nullptr;
    int64_t in_row_stride = 0;
    if (plane) {
        // s8 bytes, compact: the quantizer (or the pre-quantized copy) carries
        // the shift. The zero point moves into the *byte* domain too — a u8
        // source's `q − 128` is `q + d` — or the quantizer would clamp a
        // u8-range zero against the s8 clamp.
        int8_t* p = reinterpret_cast<int8_t*>(base + sc.quant_off);
        materialize_plane<TAct>(input, p, s_x, zp_x + d, ctx);
        in_base = reinterpret_cast<const uint8_t*>(p);
        in_row_stride = IW * IC;
        xr = 0;
    } else {
        in_base = reinterpret_cast<const uint8_t*>(input.ptr<TAct>());
        in_row_stride = input.row_stride_elems();
    }
    const int64_t ib_step = IH * in_row_stride;

    const int8_t* const w_ptr = weight.ptr<int8_t>();
    [[maybe_unused]] const TOut* b_ptr = (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    [[maybe_unused]] const int32_t* b_i32 =
        (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    TOut* const out_ptr = output.ptr<TOut>();
    [[maybe_unused]] const QuantParams& qc = output.quant_params();

    // ---- block geometry ----------------------------------------------------
    const int64_t kdim_max = pg.kdim_max;
    const bool a_biased = arch::mma_i8_folds_a_bias;
    int8_t* const shared_ptr = reinterpret_cast<int8_t*>(base + sc.panel_off);
    const bool shared = plan.shared_elems > 0;
    // The panels every task reads in place: the caller's blob, or this call's
    // own copy (null when the weight did not fit — each task packs its slot).
    const int8_t* const panels =
        (!use_sme && (prepacked || shared)) ? (prepacked ? w_ptr : shared_ptr) : nullptr;

    // ---- the shared weight prepack (once per call) -------------------------
    // Through the entry prepackWeights uses, so the addressing is one code path.
    // It runs before the sums are read back below. The SME route packs its own
    // B layout per task, so this NEON copy is skipped there.
    if (shared && !use_sme) {
        const int64_t cols = (plan.shared_elems + OC - 1) / OC;
        const int64_t shape[] = {OC, cols};
        TensorView blob(std::span<const int64_t>(shape, 2), weight.data_type(), shared_ptr, cols,
                        TensorLayout::PackedHWIO);
        conv_packed_weight_pack(attrs, weight, blob, ctx);
    }

    // R_b[oc] = Σ_k weight[oc,k] over the whole K. A packed blob carries it, so
    // the walk below is only for a filter too large for the arena, and it is
    // serial because a second pool dispatch costs more than the pass.
    //
    // In HWIO one output channel's K values are a *strided* column, so the sum
    // accumulates over the k rows — elementwise across OC, the contiguous axis.
    PoolArray<int32_t> r_b_all;
    if (col_term) {
        r_b_all.assign(static_cast<size_t>(OC), 0);
        if (panels) {
            std::memcpy(r_b_all.data(), panels + pg.sums_off,
                        static_cast<size_t>(OC) * sizeof(int32_t));
        } else {
            for (int t = 0; t < karea; ++t) {
                for (int64_t i = 0; i < ic_per_group; ++i) {
                    const int8_t* const row = w_ptr + t * w_tap_stride + i * w_ic_stride;
                    for (int64_t oc = 0; oc < OC; ++oc) {
                        r_b_all[static_cast<size_t>(oc)] += row[oc];
                    }
                }
            }
        }
    }

#if defined(NNOPS_ARCH_SME)
    // ---- the shared SME filter prepack (once per call) ----------------------
    // Per (group, task oc start, ic block), the same layout the fp route uses.
    // `ocn_block` need not be a 64-column multiple, so the region is cut at the
    // tasks' own oc steps rather than at chunk boundaries.
    std::vector<int64_t> sme_start_prefix;
    PoolArray<int8_t> sme_b_shared;
    PoolArray<int8_t> sme_stage_shared;
    int64_t sme_total_starts = 0;
    int64_t sme_region_bytes = 0;
    int64_t sme_slot_max = 0;
    int64_t sme_num_ic_blocks = 0;
    if (use_sme) {
        const int64_t nr = aarch64::sme_nr_max;
        sme_num_ic_blocks = (ic_per_group + icn_block - 1) / icn_block;
        sme_start_prefix.assign(static_cast<size_t>(oc_chunks) + 1, 0);
        for (int64_t c = 0; c < oc_chunks; ++c) {
            const int64_t lo = c * pg.panels_oc / oc_chunks * nr_max;
            const int64_t hi =
                std::min((c + 1) * pg.panels_oc / oc_chunks * nr_max, oc_per_group);
            const int64_t starts = (hi - lo + ocn_block - 1) / ocn_block;
            sme_start_prefix[static_cast<size_t>(c) + 1] =
                sme_start_prefix[static_cast<size_t>(c)] + starts;
        }
        sme_total_starts = sme_start_prefix[static_cast<size_t>(oc_chunks)];
        sme_slot_max = sme_b_panel_stride(DataType::s8, kdim_max);
        const int64_t max_blocks = (ocn_block + nr - 1) / nr;
        sme_region_bytes = max_blocks * sme_slot_max;
        const int64_t slices = G * sme_total_starts * sme_num_ic_blocks;
        sme_b_shared.reset(static_cast<size_t>(slices * sme_region_bytes));
        if (karea > 1) {
            sme_stage_shared.reset(static_cast<size_t>(kdim_max * nr));
        }
        for (int64_t g = 0; g < G; ++g) {
            const int8_t* const w_group = w_ptr + g * oc_per_group;
            for (int64_t c = 0; c < oc_chunks; ++c) {
                const int64_t oc_lo = c * pg.panels_oc / oc_chunks * nr_max;
                const int64_t oc_hi =
                    std::min((c + 1) * pg.panels_oc / oc_chunks * nr_max, oc_per_group);
                int64_t si = 0;
                for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block, ++si) {
                    const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);
                    for (int64_t ib = 0; ib < sme_num_ic_blocks; ++ib) {
                        const int64_t icn0 = ib * icn_block;
                        const int64_t icnc = std::min(icn_block, ic_per_group - icn0);
                        const int64_t kdim = icnc * karea;
                        int8_t* const dst =
                            sme_b_shared.data() +
                            ((g * sme_total_starts + sme_start_prefix[static_cast<size_t>(c)] +
                              si) *
                                 sme_num_ic_blocks +
                             ib) *
                                sme_region_bytes;
                        for (int64_t p0 = 0; p0 < ocnc; p0 += nr) {
                            const int64_t np = std::min(nr, ocnc - p0);
                            const int8_t* const bsrc =
                                w_group + ocn + p0 + icn0 * w_ic_stride;
                            int8_t* const out = dst + (p0 / nr) * sme_slot_max;
                            if (karea == 1) {
                                tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out,
                                                     bsrc, static_cast<int>(w_ic_stride));
                            } else {
                                for (int t = 0; t < karea; ++t) {
                                    for (int64_t i = 0; i < icnc; ++i) {
                                        std::memcpy(
                                            sme_stage_shared.data() +
                                                (static_cast<int64_t>(t) * icnc + i) * nr,
                                            bsrc + t * w_tap_stride + i * w_ic_stride,
                                            static_cast<size_t>(np));
                                    }
                                }
                                tile_pack_rhs_i8_sme(false, static_cast<int>(np), kdim, out,
                                                     sme_stage_shared.data(),
                                                     static_cast<int>(nr));
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    const int64_t NG = N * G;
    const int64_t chunks_per_ng = oh_chunks * oc_chunks;
    const int64_t total = NG * chunks_per_ng;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < total;

    const auto run_task = [&](int64_t t) {
        const int64_t ng = t / chunks_per_ng;
        const int64_t rem = t % chunks_per_ng;
        const int64_t chunk = rem / oc_chunks;
        const int64_t oc_chunk = rem % oc_chunks;
        const int64_t n = ng / G;
        const int64_t g = ng % G;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;
        char* const slot_ptr = base + sc.head + slot * sc.slot_stride;
        int8_t* const a_pack = reinterpret_cast<int8_t*>(slot_ptr + sc.off_a);
        int8_t* const a_buf = a_pack + plan.a_pack_stride;
        int8_t* const stage = reinterpret_cast<int8_t*>(slot_ptr + sc.off_stage);
        int8_t* const pack_buf = reinterpret_cast<int8_t*>(slot_ptr + sc.off_pack);
        int32_t* const acc_buf = reinterpret_cast<int32_t*>(
            base + sc.off_acc + slot * plan.acc_size * static_cast<int64_t>(sizeof(int32_t)));

        // The SME route's per-task A slab; B is the shared prepack.
#if defined(NNOPS_ARCH_SME)
        PoolArray<int8_t> sme_a;
        if (use_sme) {
            const int64_t kd4 = align4(kdim_max);
            const int64_t a_blocks = (m_block + aarch64::sme_mr - 1) / aarch64::sme_mr;
            sme_a.reset(static_cast<size_t>(
                a_blocks * sme_a_panel_stride(DataType::s8, kd4)));
        }
#endif

        TOut* const out_g = out_ptr + n * ob_step + g * oc_per_group;
        const uint8_t* const in_g = in_base + n * ib_step + g * ic_per_group;
        const int8_t* const w_g = w_ptr + g * oc_per_group;
        [[maybe_unused]] const TOut* const bias_g = b_ptr ? b_ptr + g * oc_per_group : nullptr;
        [[maybe_unused]] const int32_t* const ibias_g = b_i32 ? b_i32 + g * oc_per_group : nullptr;

        const int64_t r0 = chunk * OH / oh_chunks;
        const int64_t r1 = (chunk + 1) * OH / oh_chunks;
        // Over *panels*, like the fp route: a task starts where a panel does.
        const int64_t oc_lo = oc_chunk * pg.panels_oc / oc_chunks * nr_max;
        const int64_t oc_hi =
            std::min((oc_chunk + 1) * pg.panels_oc / oc_chunks * nr_max, oc_per_group);

        // Same flat-pixel rule as the fp route; pointwise also needs the input
        // dense, since it reads A straight out of the row.
        const bool flat =
            out_row_stride == OW * OC && (!plan.pointwise || in_row_stride == IW * IC);

        // Same block-end clamp and step as the fp route: the blob holds one oc
        // block at a time and the kernel walks its panels at `slot` apart.
        const auto block_end = [&](int64_t ocn) {
            return panels ? std::min(oc_hi, (ocn / ocn_block + 1) * ocn_block) : oc_hi;
        };
        for (int64_t ocn = oc_lo; ocn < oc_hi; ocn = panels ? block_end(ocn) : ocn + ocn_block) {
            const int64_t oc_end = block_end(ocn);
            const int64_t ocnc = std::min(ocn_block, oc_end - ocn);

            for (int64_t pr = r0; pr < r1; pr += ph_block) {
                const int64_t pr1 = std::min(r1, pr + ph_block);
                const int64_t p_first = pr * OW;
                const int64_t p_last = pr1 * OW;
                // Re-split a tail shorter than MR evenly, keeping the step a
                // multiple of MR so the re-split costs no extra block.
                constexpr int64_t mr_step = MR_MAX_I8;
                const int64_t p_span = std::max<int64_t>(p_last - p_first, 1);
                const int64_t p_tail = p_span % m_block;
                const int64_t p_step =
                    (flat && p_tail != 0 && p_tail < mr_step && m_block >= 2 * mr_step) ?
                        (m_block / mr_step - 1) * mr_step :
                        m_block;

                for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                    const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                    const bool is_last_icn = (icn + icn_block) >= ic_per_group;
                    const int kdim = static_cast<int>(icnc * karea);
                    const int kdim4 = static_cast<int>(align4(kdim));
                    // Pointwise reads the pixels straight out of the row, which needs
                    // a whole number of 4-byte k groups; else the gather zero-fills.
                    const bool direct_a = plan.pointwise && (kdim == kdim4);

                    const int8_t* b_panel;
                    bool b_sme = false;
#if defined(NNOPS_ARCH_SME)
                    if (use_sme) {
                        const int64_t oc_lo_chunk =
                            oc_chunk * pg.panels_oc / oc_chunks * nr_max;
                        const int64_t starts_before =
                            sme_start_prefix[static_cast<size_t>(oc_chunk)];
                        const int64_t si_local = (ocn - oc_lo_chunk) / ocn_block;
                        const int64_t slice =
                            (g * sme_total_starts + starts_before + si_local) *
                                sme_num_ic_blocks +
                            icn / icn_block;
                        b_panel = sme_b_shared.data() + slice * sme_region_bytes;
                        b_sme = true;
                    }
#endif
                    if (!b_sme) {
                        if (panels) {
                            b_panel = panels + pg.offset(g, icn / icn_block, ocn);
                        } else {
                            pack_block_i8(pack_buf, gemm_panel_slot(kdim, nr_max, 1, true),
                                          stage, w_g + ocn + icn * w_ic_stride, w_ic_stride,
                                          w_tap_stride, ocnc, icnc, karea, nr_max, kdim,
                                          a_biased);
                            b_panel = pack_buf;
                        }
                    }

                    // `mc` lives outside the loop: it is the step, and the body's
                    // early exits still have to advance it.
                    int mc = 0;
                    for (int64_t p = p_first; p < p_last; p += flat ? p_step : mc) {
                        const int64_t oh = p / OW;
                        const int64_t ow0 = p - oh * OW;
                        const int64_t left = flat ? std::min(p_step, p_last - p) : (OW - ow0);
                        mc = static_cast<int>(std::min<int64_t>(m_block, left));
                        {
                            const uint8_t* const in_row = in_g + oh * in_row_stride;

                            const int8_t* a;
                            int lda;
                            if (direct_a) {
                                // The pixel is its own patch: A is the input run.
                                a = reinterpret_cast<const int8_t*>(in_row + ow0 * IC + icn);
                                lda = static_cast<int>(IC);
                            } else {
                                tiled_im2col_i8(
                                    a_buf, kdim4, in_g + icn, in_row_stride, IC, xr,
                                    static_cast<uint8_t>(plane ? zp_x + d : zp_x), p,
                                    static_cast<int>(OW), flat, mc, static_cast<int>(IH),
                                    static_cast<int>(IW), KH, KW, SH, SW, DH, DW, PH, PW,
                                    static_cast<int>(icnc), karea, kdim, kdim4);
                                a = a_buf;
                                lda = kdim4;
                            }

                            int32_t* const acc_tile = acc_buf + (p - p_first) * ocn_block;

                            // The MMA wants A packed (group-major, MR rows of 4 bytes
                            // per k group): its direct-A variant costs ~2.7x the uops
                            // broadcasting each group out of a row-major row.
                            //
                            // The first K block stores instead of accumulating, so the
                            // tile needs no zero fill — a whole-tile write that only
                            // existed to be read back is skipped.
                            bool mma_sme = false;
#if defined(NNOPS_ARCH_SME)
                            if (use_sme) {
                                tile_pack_lhs_i8_sme(false, mc, kdim4, sme_a.data(), a, lda,
                                                     a_biased);
                                tile_mma_pack_i8_sme(
                                    mc, static_cast<int>(ocnc), kdim4, acc_tile,
                                    static_cast<int>(ocn_block), sme_a.data(), b_panel,
                                    /*block_stride=*/sme_slot_max, INT32_MIN, INT32_MAX,
                                    /*zero_mode=*/icn == 0);
                                mma_sme = true;
                            }
#endif
                            if (!mma_sme) {
                                const int64_t ldd_a = geom_ldd_i8_a(kdim4);
                                tile_pack_lhs_i8(false, mc, kdim4, a_pack,
                                                 static_cast<int>(ldd_a), a, lda, a_biased);
                                tile_mma_pack_i8(mc, static_cast<int>(ocnc), kdim4, acc_tile,
                                                 static_cast<int>(ocn_block), a_pack, b_panel,
                                                 /*ldb=*/0, INT32_MIN, INT32_MAX,
                                                 /*zero_mode=*/icn == 0);
                            }

                            if (!is_last_icn) {
                                continue;
                            }

                            // The per-pixel sums the compensation reads: over the
                            // whole K, of the *true* activation values (the gather's
                            // shift comes back out), recomputed rather than buffered.
                            int32_t ra_tile[64];
                            const int32_t* ra_ptr = nullptr;
                            if (comp) {
                                NNOPS_ASSERT(mc <= 64);
                                if (direct_a) {
                                    // The pixel is its own patch, its channels one
                                    // contiguous run: a row sum of the bytes the MMA
                                    // reads — no gather, no A tile.
                                    for (int m = 0; m < mc; ++m) {
                                        ra_tile[m] = k::int8_row_sum<int8_t>(
                                            reinterpret_cast<const int8_t*>(in_row) +
                                                static_cast<int64_t>(ow0 + m) * IC,
                                            static_cast<int>(ic_per_group));
                                    }
                                } else {
                                    for (int m = 0; m < mc; ++m) {
                                        ra_tile[m] = 0;
                                    }
                                    for (int64_t icn2 = 0; icn2 < ic_per_group; icn2 += icn_block) {
                                        const int64_t icnc2 =
                                            std::min(icn_block, ic_per_group - icn2);
                                        const int kd2 = static_cast<int>(icnc2 * karea);
                                        const int kd24 = static_cast<int>(align4(kd2));
                                        tiled_im2col_i8(
                                            a_buf, kd24, in_g + icn2, in_row_stride, IC, xr,
                                            static_cast<uint8_t>(plane ? zp_x + d : zp_x), p,
                                            static_cast<int>(OW), flat, mc, static_cast<int>(IH),
                                            static_cast<int>(IW), KH, KW, SH, SW, DH, DW, PH, PW,
                                            static_cast<int>(icnc2), karea, kd2, kd24);
                                        for (int m = 0; m < mc; ++m) {
                                            ra_tile[m] += k::int8_row_sum<int8_t>(
                                                a_buf + static_cast<int64_t>(m) * kd24, kd2);
                                        }
                                    }
                                }
                                const int32_t shift_total = d * static_cast<int32_t>(k_full);
                                for (int m = 0; m < mc; ++m) {
                                    ra_tile[m] -= shift_total;
                                }
                                ra_ptr = ra_tile;
                            }

                            TOut* const dst = out_g + oh * out_row_stride + ow0 * OC + ocn;
                            if constexpr (kIntOut) {
                                Int8EpilogueParams ip{};
                                ip.r_a = ra_ptr;
                                ip.s_a = nullptr; // per-tensor activation
                                ip.zp_a = nullptr;
                                ip.r_b =
                                    col_term ? r_b_all.data() + g * oc_per_group + ocn : nullptr;
                                ip.s_b = qw.scale_data ? qw.scale_data + g * oc_per_group + ocn :
                                                         nullptr;
                                ip.zp_b = qw.zero_point_data ?
                                              qw.zero_point_data + g * oc_per_group + ocn :
                                              nullptr;
                                ip.s_a_scalar = s_x;
                                ip.s_b_scalar = qw.scale;
                                ip.zp_a_scalar = zp_x;
                                ip.zp_b_scalar = qw.zero_point;
                                ip.a_bias_offset = d;
                                ip.k = static_cast<int32_t>(k_full);
                                ip.relu = attrs.epilogue.type == EpilogueActivateType::Relu;
                                ip.compensate = comp;
                                ip.col_term = col_term;
                                ip.out_scale = quant_scale_at(qc, 0);
                                ip.zp_out = quant_zp_at(qc, 0);
                                if (attrs.epilogue.quant == EpilogueQuantType::Requantize &&
                                    attrs.epilogue.quant_param_count > 0) {
                                    ip.row_out_scales = attrs.epilogue.quant_scales;
                                    ip.row_out_zps = attrs.epilogue.quant_zero_points;
                                    ip.out_scale = epilogue_quant_scale(attrs.epilogue, 0);
                                    ip.zp_out = epilogue_quant_zero(attrs.epilogue, 0);
                                }
                                ip.bias = nullptr; // the bias is per column
                                ip.col_bias = ibias_g ? ibias_g + ocn : nullptr;
                                if constexpr (std::is_same_v<TOut, uint8_t>) {
                                    ip.out_u8 = dst;
                                } else {
                                    ip.out = dst;
                                }
                                ip.ldc_out = static_cast<int>(OC);
                                int8_epilogue_tile(ip, acc_tile, static_cast<int>(ocn_block), 0, 0,
                                                   mc, ocnc);
                            } else {
                                DqEpilogueParams<TOut> ip{};
                                ip.r_a = ra_ptr;
                                ip.s_a = nullptr;
                                ip.zp_a = nullptr;
                                ip.r_b =
                                    col_term ? r_b_all.data() + g * oc_per_group + ocn : nullptr;
                                ip.s_b = qw.scale_data ? qw.scale_data + g * oc_per_group + ocn :
                                                         nullptr;
                                ip.zp_b = qw.zero_point_data ?
                                              qw.zero_point_data + g * oc_per_group + ocn :
                                              nullptr;
                                ip.row_off = nullptr;
                                ip.s_a_scalar = s_x;
                                ip.s_b_scalar = qw.scale;
                                ip.zp_a_scalar = zp_x;
                                ip.zp_b_scalar = qw.zero_point;
                                ip.a_bias_offset = d;
                                ip.k = static_cast<int32_t>(k_full);
                                ip.compensate = comp;
                                ip.col_term = col_term;
                                ip.act_u8 = u8a;
                                ip.ep = &attrs.epilogue;
                                ip.bias = nullptr;
                                ip.col_bias = bias_g ? bias_g + ocn : nullptr;
                                ip.out = dst;
                                ip.ldc_out = static_cast<int>(OC);
                                dq_epilogue_tile<TOut>(ip, acc_tile, static_cast<int>(ocn_block), 0,
                                                       0, mc, ocnc);
                            }
                        }
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, total, run_task);
}

// =========================================================================
//  Plan
// =========================================================================

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

    if (quant) {
        // Activation: pre-quantized (u8/s8), or fp (quantized per call).
        if (dt != DataType::u8 && dt != DataType::s8 && dt != DataType::f32 &&
            dt != DataType::f16) {
            return plan;
        }
        const bool int_out = out.dtype == DataType::s8 || out.dtype == DataType::u8;
        if (!int_out && out.dtype != DataType::f32 && out.dtype != DataType::f16) {
            return plan;
        }
        if (!int_out && (dt == DataType::f32 || dt == DataType::f16) && out.dtype != dt) {
            return plan;
        }
        const DataType bias_dt = int_out ? DataType::s32 : out.dtype;
        if (inputs.size() > 2 && inputs[2].dtype != bias_dt) {
            return plan;
        }
        // Only the reference implements add_to's fold on the quantized route.
        if (attrs.add_to) {
            return plan;
        }
    } else {
        if (!is_fp_activation_dtype(dt)) {
            return plan;
        }
        if (wt.dtype != dt || out.dtype != dt) {
            return plan;
        }
        if (inputs.size() > 2 && inputs[2].dtype != dt) {
            return plan;
        }
    }

    const int64_t N = in.dims[0];
    const int64_t IC = in.dims[1];
    // The weight is TensorFlow's [KH, KW, IC/G, OC]; anything else
    // is not this route's — except the prepacked blob, whose shape leads with OC
    // (see conv_packed_weight_desc).
    const bool sme_blob = wt.layout == TensorLayout::PackedHWIOSme;
    const bool prepacked = is_packed_conv_weight(wt.layout);
    if (!is_channel_last_weight(wt.layout) && !prepacked) {
        return plan;
    }
    const int64_t OC = prepacked ? wt.dims[0] : wt.dims[3];
    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];

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
    plan.m_block = sme_route ? 2 * aarch64::sme_mr : kMBlock;
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
    // call. A pre-quantized u8 one is materialized too
    // when it takes the pointwise route, whose in-row A read cannot shift bytes.
    const bool fp_act = (dt == DataType::f32 || dt == DataType::f16);
    plan.quant_size = (quant && (fp_act || (dt == DataType::u8 && plan.pointwise))) ?
                          N * in.dims[2] * in.dims[3] * IC :
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

/// [N, C, H, W] cut to `len` columns from `col0`, pitch and layout kept: a
/// column is a whole channel vector.
TensorView wide_window(const TensorView& v, int64_t col0, int64_t len)
{
    const int64_t unit = static_cast<int64_t>(data_type_stride_unit(v.data_type()));
    const int64_t off = col0 * v.shape(1) * unit;
    const int64_t s[4] = {v.shape(0), v.shape(1), v.shape(2), len};
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
    const int64_t OW = output.shape(3);
    const int64_t IW = input.shape(3);
    const int64_t workers = std::max<int64_t>(ctx.cpu.worker_count(), 1);
    if (output.shape(2) != 1 || workers <= 1 || inputs.size() > 4) {
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
        const int64_t OW = output.shape(3);
        const int64_t IW = inputs[0].shape(3);
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
    g.icn_block =
        unit_1x1 ? ic_per_group :
                   std::min<int64_t>(impl::get_ic_block(karea, ic_per_group),
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
    if (ic_per_group <= 0 || oc_per_group <= 0 || attrs.groups <= 0 ||
        (weight_dtype != DataType::f32 && weight_dtype != DataType::f16)) {
        return g;
    }

    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];
    const bool unit_1x1 = karea == 1 && attrs.stride[0] == 1 && attrs.stride[1] == 1;
    // The float ZA kernel's n tile: 2 * 16 columns. A constant here so this
    // geometry stays buildable without the SME headers.
    constexpr int64_t kSmeBlockN = 32;
    g.sme = true;
    g.nr = kSmeBlockN;
    g.icn_block =
        unit_1x1 ? ic_per_group :
                   std::min<int64_t>(impl::get_ic_block(karea, ic_per_group),
                                     ic_per_group);
    // The task N tile is a whole number of 32-column ZA blocks.
    const int64_t oc_tile = std::min<int64_t>(oc_per_group, impl::kOcnBlock);
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
    const int64_t sme_oc_min = aarch64::sme_fp_nr;
    const bool sme_blob = (weight.dtype == DataType::f32 || weight.dtype == DataType::f16) &&
                          (OC / G) >= sme_oc_min && aarch64::available_sme();
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
                    tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out, bsrc,
                                      static_cast<int>(w_ic));
                } else {
                    for (int t = 0; t < karea; ++t) {
                        for (int64_t i = 0; i < icnc; ++i) {
                            std::memcpy(stage.data() +
                                            (static_cast<int64_t>(t) * icnc + i) * nr,
                                        bsrc + t * w_tap + i * w_ic,
                                        static_cast<size_t>(np) * sizeof(T));
                        }
                    }
                    tile_pack_rhs_sme(false, static_cast<int>(np), kdim, out, stage.data(),
                                      static_cast<int>(nr));
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
        } else {
            NNOPS_ASSERT(dtype == DataType::f16);
            pack_weight_sme<half>(weight, packed, g, ICG, OCG, G, ctx);
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
