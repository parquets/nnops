/// @file conv2d_implicit_gemm.cpp
/// @brief Conv2D as implicit GEMM — both memory layouts, one driver.
///
/// conv2d_implicit_gemm.h describes the two mappings and everything they share
/// (one plan, one scratch geometry, one entry). This file is the two routes:
///
///   planar       — NCHW, M = output channels, N = pixels. The activations are
///                  gathered into packed B; the raw weight is the direct-A
///                  operand. The fp route fuses the gather with the pack
///                  (tiled_im2col_2d_packed, so no col matrix exists at all);
///                  the quantized one gathers bytes into a col buffer and packs
///                  afterwards, with an int32 accumulator and a packed weight.
///   channel_last — NHWC, M = pixels, N = oc block, K tap-major. The gathered
///                  activation tile is A and the packed HWIO weight is B, either
///                  prepacked once per call (shared arena region) or per task.
///
/// The route comes from the activation's layout in conv2d_implicit_gemm_kernel,
/// which owns the plan/pool/dispatch prologue and hands each route its plan.
/// Each route opens its own anonymous namespace, so helpers (tile_save,
/// pack_block, …) carry no route prefix. The decomposition is the same on both:
/// (batch, group) × oh-chunks × oc-chunks. Output tiles are disjoint, so every
/// route is bit-identical to serial execution.
///
/// What not to "fix" later:
///
///  * The first icn block BUILDS the tile — zero_mode, or the bias as the
///    accumulator's initial value (tile_mma_direct's per-row bias on the planar
///    route, its per-column col_bias on channel-last, whose lanes are output
///    channels). Seeding the tile first costs a write plus a read-back of the
///    whole tile for one addend. Only the block that builds it may be seeded: a
///    later one would re-seed and drop the partial sum. add_to is the exception
///    — it folds the *original* output, which no accumulator seed can express.
///  * A pure Relu/Relu6 folds into the MMA store's min/max, on the last icn
///    block only (partial sums accumulate unclamped). Measured 11% of the conv2d
///    row at one thread, bit-identical checkpoints. add_to adds the residual
///    *after* the activation, so it keeps the separate epilogue pass.
///  * The prepacked filter's geometry comes from conv_packed_weight_last /
///    conv_packed_weight_planar, and the plan ADOPTS its icn_block/ocn_block
///    from there instead of choosing again: prepackWeights built the panels
///    against those functions, with no activation to resolve a plan from. A plan
///    that picked its own block would address panels the blob does not hold.
///  * An fp activation is quantized into a compact plane once per call, never
///    inside the gather: the gather repeats once per oc block, and only the
///    materialized plane lets a 1x1 conv take the pointwise route.
///
/// Plan heuristics, measured at one thread unless stated (this host, 6P+8E):
///
///  * ic-block ladder: kdim (= ic_block × karea) in ~144–200 is the cost knob;
///    past it the col buffer the block multiplies costs more than the passes it
///    saves. Every ic block pays a full read-modify-write of its output tile and
///    restarts the gather. A 3x3 at the 8 rung (kdim 72) is 2.5–4.4% behind the
///    16 rung; the wider filters move much less.
///  * The oh-block starts at align(PH+1, 4). It halves while more than half the
///    pool is idle (1x256x28x28 3x3, 7→14 blocks, 406→466 GFLOPS), flooring at
///    two n-panels so the weight panel is not re-read every oh-step. A mirror
///    loop grows it when the filter count outnumbers the plane area — the weak
///    filter-vs-plane test, not panel bytes vs the plane, which would also catch
///    1x1 planes the halving already keeps at 4 rows — and stops once the next
///    size would leave more than half the pool idle. Separately it grows back
///    while every worker still gets a whole block of its own: +10..15% at one
///    thread on the 28-56 row planes, a wash at 112 where the scratch has caught
///    up. That grown block count must stay a multiple of the chunk count (an
///    uneven split cost −12% at 4 threads on a 112 plane) and must not make the
///    busiest chunk longer (three 30-row chunks against a 22-row one cost −4.8%
///    at 4 threads); the scratch cap is 256K elements a slot.
///  * oc-chunks use the nearest whole number to nt/plane_tasks, not the ceiling:
///    a 9-block plane at 20 threads wants 2 chunks, ceil picks 3 and the barrier
///    waits on a 7-task second wave (152 vs 226 GFLOPS, 17x17). Planar splits
///    the filter axis only on fp (its quantized accumulator is per oh-chunk);
///    channel-last splits it on both, with its accumulator per pass.
///  * channel-last adds a rows-per-pass cap (ph_block, 128 KiB of accumulator):
///    a 56x56 full-tile set is 800 KB and every pass went to DRAM. Pixel tiles
///    flatten across row boundaries when rows are dense, so a narrow plane stops
///    paying MR-row granularity per row. The shared prepack cap is 8 MiB; past
///    it each task packs its own panels.
///  * The quantized routes' R_a/R_b walk over the weight is deliberately serial:
///    a second pool dispatch costs more than the pass (1.18x at 2T → 1.9x at 20T
///    on a 3x3 whose zero points are not the packer's).

#include "conv2d_implicit_gemm.h"
#include "backend/cpu/matmul/matmul_helper.h"               // uniform pack/MMA + panel strides
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"        // epilogue / dequant epilogues
#include "backend/cpu/simd_kernel/simd_im2col.hpp"          // tiled im2col gathers
#include "backend/cpu/simd_kernel/simd_int8_reduce.hpp"     // int8_row_sum / int8_col_sum_slab
#include "backend/cpu/simd_kernel/simd_quant.hpp"           // quant_kernel::quantization
#include "backend/cpu/common/memory_pool.hpp"               // internal scratch-memory pool
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

// =========================================================================
//  Planar route (NCHW) — im2col into packed B, raw weight as direct A
// =========================================================================

namespace planar {
namespace {

using namespace nnops::simd;
namespace k = nnops::kernel;  // pure SIMD kernels: tiled_im2col_2d / tiled_im2col_3d

// ---- block-size heuristics (ported from nn_compute) ----------------------

/// Input-channel block: kdim = ic_block × karea lands in ~144–200.
inline int get_ic_block(int karea) noexcept {
    if (karea == 1)  { return 64; }
    if (karea <= 9)  { return 16; }
    if (karea <= 25) { return 8;  }
    if (karea <= 49) { return 2;  }
    return 1;
}

constexpr int kOcnBlock = 128;  // output-channel tile (nn_compute single-thread default)
constexpr int kAlign     = 4;   // oh-block alignment

// ---- tile init / save / add helpers --------------------------------------

/// Fill a tile with the per-row bias, or add it into what is already there.
template <class T>
void tile_set_bias(T* NNOPS_RESTRICT tile, int ld, int M, int N,
                   const T* NNOPS_RESTRICT bias, bool add_to) noexcept
{
    constexpr int L = simd_lane_for<T>;
    for (int m = 0; m < M; ++m) {
        T* row = tile + m * ld;
        const float b = bias ? s_load(&bias[m]) : 0.0f;
        const auto vb = v_set1(row, b);
        int i = 0;
        if (add_to) {
            for (; i + L <= N; i += L) {
                v_store(row + i, v_add(v_load(row + i), vb));
            }
            for (; i < N; ++i) {
                s_store(&row[i], s_load(&row[i]) + b);
            }
        } else {
            for (; i + L <= N; i += L) {
                v_store(row + i, vb);
            }
            for (; i < N; ++i) {
                s_store(&row[i], b);
            }
        }
    }
}

/// Copy a strided tile (row stride ld) into a dense scratch buffer (row stride N).
template <class T>
void tile_save(T* NNOPS_RESTRICT dst, const T* NNOPS_RESTRICT src,
               int ld, int M, int N) noexcept
{
    constexpr int L = simd_lane_for<T>;
    for (int m = 0; m < M; ++m) {
        const T* s = src + m * ld;
        T* d = dst + m * N;
        int i = 0;
        for (; i + L <= N; i += L) {
            v_store(d + i, v_load(s + i));
        }
        for (; i < N; ++i) {
            d[i] = s[i];
        }
    }
}

// ---- main kernel ----------------------------------------------------------

template <class T>
void conv2d_planar_impl(const Conv2DAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        const Conv2DGemmPlan& plan,
                        void* workspace)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // Weight: [OC, IC/G, KH, KW]
    const int64_t OC = weight.shape(0);

    // Output: [N, OC, OH, OW]
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];
    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    const int64_t karea = KH * KW;

    const int64_t icn_block = plan.icn_block;
    const int64_t ocn_block = plan.ocn_block;
    const int64_t oh_block  = plan.oh_block;
    const int64_t num_oh_blocks = plan.num_oh_blocks;
    const int64_t oh_chunks     = plan.oh_chunks;
    const int64_t oc_chunks     = plan.oc_chunks;

    // Strides (elements).
    const int64_t icn_step = IH * in_row_stride;         // between input channels
    const int64_t ib_step  = IC * icn_step;              // between batch samples
    const int64_t ig_step  = ic_per_group * icn_step;    // between groups (input)

    const int64_t ocn_step = OH * out_row_stride;        // between output channels
    const int64_t ob_step  = OC * ocn_step;              // between batch samples
    const int64_t og_step  = oc_per_group * ocn_step;    // between groups (output)

    // Weight row stride within a group: [oc_per_group][ic_per_group][karea].
    // Dims, not pitch: a weight is a load-time constant.
    const int64_t w_ocn_step = ic_per_group * karea;

    T* out_ptr = output.ptr<T>();
    const T* in_ptr  = input.ptr<T>();
    const T* w_ptr   = weight.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;

    // One slot per worker when the backend reports thread ids, else one per
    // task. No col buffer on this route: the gather writes packed B directly,
    // so a slot is [orig][pack_b].
    NNOPS_ASSERT(plan.col_size == 0);
    constexpr int nr_max = nr_max_flt<T>();
    constexpr int esz    = static_cast<int>(sizeof(T));
    const Conv2DGemmScratch sc = gemm_scratch_layout(plan, esz);
    char* const arena = static_cast<char*>(workspace);

    const int64_t NG = N * G;
    const int64_t chunks_per_ng = plan.oh_chunks * plan.oc_chunks;
    const int64_t total = NG * chunks_per_ng;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < total;

    const bool epilogue_active = attrs.epilogue.type != EpilogueActivateType::None;

    // A pure clamp needs no epilogue pass: the MMA store's min/max bounds carry
    // it. add_to is the exception — it adds the residual *after* the
    // activation, so it needs the tile back in memory.
    constexpr float kInf = std::numeric_limits<float>::infinity();
    float clamp_lo = -kInf;
    float clamp_hi = kInf;
    bool epilogue_folded = false;
    if (!attrs.add_to) {
        switch (attrs.epilogue.type) {
        case EpilogueActivateType::Relu:  clamp_lo = 0.0f; epilogue_folded = true; break;
        case EpilogueActivateType::Relu6: clamp_lo = 0.0f; clamp_hi = 6.0f;
                                          epilogue_folded = true; break;
        default: break;
        }
    }
    const bool epilogue_pass = epilogue_active && !epilogue_folded;

    // The first icn block builds the tile from the bias row (tile_mma_direct's
    // bias argument: the kernel's row index is the output channel, so it lands
    // exactly where tile_set_bias would have written). add_to's original
    // output is the exception — it still pre-fills.
    const bool bias_in_mma = has_bias && !attrs.add_to;
    const bool seed_needed = attrs.add_to;

    const auto run_task = [&](int64_t t) {
        const int64_t ng      = t / chunks_per_ng;
        const int64_t rem     = t % chunks_per_ng;
        // The oc-chunk varies fastest: neighbours in task order then cover
        // different filter rows, so their weight streams differ.
        const int64_t chunk    = rem / oc_chunks;
        const int64_t oc_chunk = rem % oc_chunks;
        const int64_t n = ng / G;
        const int64_t g = ng % G;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;
        char* const slot_ptr = arena + sc.head + slot * sc.slot_stride;
        T* orig   = attrs.add_to ? reinterpret_cast<T*>(slot_ptr + sc.off_orig)
                                 : nullptr;
        T* pack_b = reinterpret_cast<T*>(slot_ptr + sc.off_pack);

        T* output_ptr = out_ptr + n * ob_step + g * og_step;
        const T* input_ptr  = in_ptr + n * ib_step + g * ig_step;
        const T* weight_ptr = w_ptr + g * oc_per_group * ic_per_group * karea;
        const T* bias_ptr   = b_ptr ? b_ptr + g * oc_per_group : nullptr;

        // This chunk's contiguous oh-block range (disjoint across chunks).
        const int64_t b0 = chunk * num_oh_blocks / oh_chunks;
        const int64_t b1 = (chunk + 1) * num_oh_blocks / oh_chunks;

        // Disjoint oc range, split evenly over filter rows.
        const int64_t oc_lo = oc_chunk * oc_per_group / oc_chunks;
        const int64_t oc_hi = (oc_chunk + 1) * oc_per_group / oc_chunks;

        for (int64_t b = b0; b < b1; ++b) {
            const int64_t oh       = b * oh_block;
            const int64_t ohc      = std::min(oh_block, OH - oh);
            const int64_t roi_area = ohc * OW;

            const int64_t ih = oh * SH - PH;
            const int64_t iw = 0 * SW - PW;
            const int roi_pad_h = (ih < 0) ? static_cast<int>(-ih)
                                           : (ih >= IH ? static_cast<int>(ih - IH) : 0);
            const int roi_pad_w = (iw < 0) ? static_cast<int>(-iw)
                                           : (iw >= IW ? static_cast<int>(iw - IW) : 0);

            T* tile_base = output_ptr + oh * out_row_stride;

            for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                const bool is_last_icn = (icn + icn_block) >= ic_per_group;

                const T* input_local = input_ptr + icn * icn_step;

                // B = [K=icnc*karea][N=roi_area], packed once per ic-block into
                // uniform nr_max-wide panels; every oc-block below reuses it.
                const int kdim = static_cast<int>(icnc * karea);
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * kdim * esz) / esz);

                if (!plan.pointwise) {
                    // Gather straight into the panels: materializing the col
                    // matrix first would move the same bytes twice for a pure
                    // re-ordering, with identical values and panel layout.
                    k::tiled_im2col_2d_packed<T, nr_max>(pack_b, ldd_b, input_local,
                                                 0, static_cast<int>(oh),
                                                 static_cast<int>(OW), static_cast<int>(ohc),
                                                 roi_pad_h, roi_pad_w,
                                                 static_cast<int>(PH), static_cast<int>(PW),
                                                 static_cast<int>(icnc), static_cast<int>(IH), static_cast<int>(IW),
                                                 static_cast<int>(icn_step), static_cast<int>(in_row_stride),
                                                 static_cast<int>(KH), static_cast<int>(KW),
                                                 static_cast<int>(SH), static_cast<int>(SW),
                                                 static_cast<int>(DH), static_cast<int>(DW));
                } else {
                    // A pointwise gather would be a dense row copy; read B from
                    // the input instead. pad == 0 and stride == 1 keep the ROI
                    // inside the input, so there is no border case. The ROI is
                    // one roi_area-wide run per channel, so k strides by the
                    // channel stride, not the ROI width.
                    const T* b_src = input_local + oh * in_row_stride;
                    tile_pack_rhs(false, static_cast<int>(roi_area), kdim,
                                  pack_b, ldd_b, b_src, static_cast<int>(icn_step), 1.0f);
                }

                for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block) {
                    const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);
                    T* tile = tile_base + ocn * ocn_step;

                    if (icn == 0 && seed_needed) {
                        // add_to keeps the original output: epilogue(bias + Σ)
                        // then += orig.
                        if (attrs.add_to && epilogue_pass) {
                            tile_save<T>(orig, tile, static_cast<int>(ocn_step),
                                         static_cast<int>(ocnc), static_cast<int>(roi_area));
                        }
                        tile_set_bias<T>(tile, static_cast<int>(ocn_step),
                                         static_cast<int>(ocnc), static_cast<int>(roi_area),
                                         bias_ptr ? bias_ptr + ocn : nullptr,
                                         attrs.add_to && !epilogue_active);
                    }

                    const T* a = weight_ptr + icn * karea + ocn * w_ocn_step;
                    // Only the final ic-block clamps; partial sums accumulate
                    // unclamped.
                    const bool clamp_here = is_last_icn && epilogue_folded;
                    tile_mma_direct(static_cast<int>(ocnc), static_cast<int>(roi_area),
                                            kdim,
                                            tile, static_cast<int>(ocn_step),
                                            a, static_cast<int>(w_ocn_step),
                                            pack_b,
                                            clamp_here ? clamp_lo : -kInf,
                                            clamp_here ? clamp_hi : kInf,
                                            /*zero_mode=*/icn == 0 && !seed_needed,
                                            (icn == 0 && bias_in_mma) ? bias_ptr + ocn : nullptr);

                    // An identity epilogue is a whole-tile copy-back; skip it.
                    if (is_last_icn && epilogue_pass) {
                        epilogue_inplace<T>(static_cast<int>(ocnc), static_cast<int>(roi_area),
                                            tile, static_cast<int>(ocn_step),
                                            static_cast<const T*>(nullptr), attrs.epilogue,
                                            orig, attrs.add_to && epilogue_active);
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, total, run_task);
}

// ---- the quantized route --------------------------------------------------
//
// Integer operands: A the s8 weight, B the u8/s8 im2col activation, int32
// accumulator; the epilogue compensates the zero points and either dequantizes
// (fp destination) or requantizes (s8/u8 destination).

/// True when every zero point the tensor carries equals @p v.
inline bool all_zp_equal(const QuantParams& q, int64_t n, int32_t v) noexcept {
    if (q.zero_point_data == nullptr) { return q.zero_point == v; }
    for (int64_t i = 0; i < n; ++i) {
        if (q.zero_point_data[i] != v) { return false; }
    }
    return true;
}

/// Quantize an fp activation into the compact int8 plane, walking the source
/// pitch.
template <class TAct, class TDst>
void quantize_plane(const TensorView& in, TDst* dst, float scale, int32_t zero,
                    const ComputeContext& ctx)
{
    const int64_t IC = in.shape(1);
    const int64_t IH = in.shape(2);
    const int64_t IW = in.shape(3);
    const int64_t N  = in.shape(0);
    const int64_t rs = in.row_stride_elems();
    const TAct* src  = in.ptr<TAct>();

    const float s = scale;
    const float z = static_cast<float>(zero);
    const int64_t total = N * IC * IH * IW;

    auto quant_range = [&](int64_t beg, int64_t end) {
        if (rs == IW) {
            k::quant_kernel::quantization<TDst>(
                static_cast<int>(1), static_cast<int>(end - beg), dst + beg, 1,
                src + beg, 1, &s, &z);
            return;
        }
        // Pitched source: row by row, never across the pitch tail.
        for (int64_t e = beg; e < end;) {
            const int64_t row = e / IW;
            const int64_t off = e - row * IW;
            const int64_t len = std::min(IW - off, end - e);
            k::quant_kernel::quantization<TDst>(
                static_cast<int>(1), static_cast<int>(len), dst + e, 1,
                src + row * rs + off, 1, &s, &z);
            e += len;
        }
    };

    // Below this a dispatch costs more than the pass: ~100 us to re-spawn the
    // pool at 4 threads against ~10 us of work for a 200k-element plane.
    constexpr int64_t kInlineMax = int64_t{1} << 19;

    const int64_t nt = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    if (nt == 1 || total <= kInlineMax) {
        quant_range(0, total);
        return;
    }
    const int64_t slab = (total + nt - 1) / nt;
    const int64_t nslab = (total + slab - 1) / slab;
    ctx.cpu.run(0, nslab, [&](int64_t i) {
        quant_range(i * slab, std::min(i * slab + slab, total));
    });
}

/// One (activation, storage, output) dtype triple of the quantized route.
template <class TAct, class TDst, class TOut>
void conv2d_planar_int8_impl(const Conv2DAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             const Conv2DGemmPlan& plan,
                             void* workspace)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    // The B packer XORs 0x80: stored bytes are read as unsigned, then shifted
    // to (value − 128).
    constexpr bool kU8 = std::is_same_v<TDst, uint8_t>;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    // Weight: [OC, IC/G, KH, KW], or its prepacked A panels (OC first either way).
    const bool prepacked = is_packed_conv_weight(weight.layout());
    const int64_t OC = weight.shape(0);

    // Output: [N, OC, OH, OW]
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G  = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

    const int64_t KH = attrs.kernel_size[0];
    const int64_t KW = attrs.kernel_size[1];
    const int64_t SH = attrs.stride[0];
    const int64_t SW = attrs.stride[1];
    const int64_t DH = attrs.dilation[0];
    const int64_t DW = attrs.dilation[1];
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    const int64_t karea = KH * KW;
    const int64_t k_full = ic_per_group * karea;   // the epilogue's K

    const int64_t icn_block = plan.icn_block;
    const int64_t ocn_block = plan.ocn_block;
    const int64_t oh_block  = plan.oh_block;
    const int64_t num_oh_blocks = plan.num_oh_blocks;
    const int64_t oh_chunks     = plan.oh_chunks;

    const int64_t ocn_step = OH * out_row_stride;        // between output channels
    const int64_t ob_step  = OC * ocn_step;              // between batch samples
    const int64_t og_step  = oc_per_group * ocn_step;    // between groups (output)

    // Weight row stride within a group: [oc_per_group][ic_per_group][karea].
    // Dims, not pitch: a weight is a load-time constant.
    const int64_t w_ocn_step = ic_per_group * karea;

    // ---- quantization parameters -----------------------------------------
    const QuantParams& qa = input.quant_params();
    const QuantParams& qw = weight.quant_params();

    // Per-tensor activation parameters: fixed offline, valid whether the
    // activation arrives quantized or as fp.
    const float   s_x    = quant_scale_at(qa, 0);
    const int32_t zp_x   = quant_zp_at(qa, 0);
    // The zero point in the shifted space the packer produces (u8 → value−128).
    const int32_t zp_eff = zp_x - (kU8 ? 128 : 0);
    NNOPS_ASSERT(s_x > 0.0f);
    NNOPS_ASSERT(!kU8 || (zp_x >= 0 && zp_x <= 255));
    NNOPS_ASSERT(kU8 || (zp_x >= -128 && zp_x <= 127));

    // Only the weight's dtype drives the A-side bias.
    const bool    a_biased      = int8_a_pack_bias(DataType::s8) != 0;
    const int32_t a_bias_offset = int8_epilogue_offset(DataType::s8);

    // The epilogue's compensation is
    //     raw − (a_bias_offset + zp_a[m])·r_b[n] − zp_b·r_a[m] + k·zp_a[m]·zp_b
    // Only the first product is per column: it survives exactly when the
    // weight's zero point is not already the compensated one. The rest is per
    // output channel — into the bias for an integer destination, into an int32
    // row offset for the fp one (row_fold / bias_fold below).
    const bool need_col_sum = !all_zp_equal(qw, OC, -a_bias_offset);
    // r_a feeds both the column term and the fold.
    const bool need_comp = need_col_sum || zp_eff != 0;

    // ---- the activation as stored bytes -----------------------------------
    // From here the fp route is indistinguishable from the pre-quantized one:
    // the gather copies bytes and the pad value is the zero point.
    const TDst* in_base = nullptr;
    int64_t in_row_stride = 0;
    if constexpr (kFpAct) {
        TDst* plane = reinterpret_cast<TDst*>(workspace);
        quantize_plane<TAct, TDst>(input, plane, s_x, zp_x, ctx);
        in_base = plane;
        in_row_stride = IW;               // the plane is compact
    } else {
        in_base = input.ptr<TDst>();      // TAct == TDst on this route
        in_row_stride = input.row_stride_elems();
    }

    const int64_t icn_step = IH * in_row_stride;         // between input channels
    const int64_t ib_step  = IC * icn_step;              // between batch samples
    const int64_t ig_step  = ic_per_group * icn_step;    // between groups (input)

    const int8_t* w_ptr = weight.ptr<int8_t>();
    // The bias is the output dtype on the fp destination, int32 on the integer one.
    [[maybe_unused]] const TOut* b_ptr =
        (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    [[maybe_unused]] const int32_t* b_i32 =
        (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    TOut* out_ptr       = output.ptr<TOut>();
    // The requant's output parameters, read only for an integer destination.
    [[maybe_unused]] const QuantParams& qc = output.quant_params();

    // R_a[oc] = Σ_k weight[oc,k] over the whole K; the global output-channel
    // index works across groups. Deliberately serial — a second pool dispatch
    // costs more than this one SIMD pass (see the header).
    // Panel geometry from the same function prepackWeights used.
    const ConvPackedWeightPlanar pg =
        conv_packed_weight_planar(attrs, ic_per_group, oc_per_group,
                                  weight.data_type());
    NNOPS_ASSERT(!prepacked ||
                 (pg.icn_block == icn_block && pg.ocn_block == ocn_block));

    std::vector<int32_t> r_a;
    if (need_comp) {
        r_a.resize(static_cast<size_t>(OC));
        if (prepacked) {
            // The blob carries the sums, so no per-call walk of MMA-layout panels.
            std::memcpy(r_a.data(), w_ptr + pg.sums_off,
                        static_cast<size_t>(OC) * sizeof(int32_t));
        } else {
            for (int64_t oc = 0; oc < OC; ++oc) {
                r_a[static_cast<size_t>(oc)] =
                    k::int8_row_sum<int8_t>(w_ptr + oc * w_ocn_step, k_full);
            }
        }
    }

    // The per-channel half of the compensation, once per call instead of once
    // per element: −zp_b·(r_a − k·zp_a) per output channel. The integer
    // destination folds it into the bias, the fp one into an int32 row offset;
    // either way the epilogue then only handles the column term.
    std::vector<int32_t> row_fold;
    std::vector<int32_t> bias_fold;
    if (!need_col_sum && zp_eff != 0) {
        row_fold.resize(static_cast<size_t>(OC));
        for (int64_t oc = 0; oc < OC; ++oc) {
            const int32_t coef_zp =
                r_a[static_cast<size_t>(oc)] -
                static_cast<int32_t>(k_full) * quant_zp_at(qw, oc);
            row_fold[static_cast<size_t>(oc)] = -zp_eff * coef_zp;
        }
        if constexpr (kIntOut) {
            bias_fold.resize(static_cast<size_t>(OC));
            for (int64_t oc = 0; oc < OC; ++oc) {
                bias_fold[static_cast<size_t>(oc)] =
                    row_fold[static_cast<size_t>(oc)] + (b_i32 ? b_i32[oc] : 0);
            }
        }
    }

    // R_b[column] = Σ_k (stored byte) over one output column's whole K,
    // accumulated across ic-blocks; per slot and per oh-block.
    const int64_t r_b_len = plan.oh_block * OW;
    std::vector<int32_t> r_b_buf(
        need_col_sum ? static_cast<size_t>(plan.num_slots * r_b_len) : 0);

    // ---- scratch slices ---------------------------------------------------
    // The activation plane sits at the front of the arena, shared across slots;
    // per-slot data follows, acc / pack_a sit behind the slots.
    // gemm_scratch_layout decides all of it.
    const Conv2DGemmScratch sc = gemm_scratch_layout(plan, 1);
    char* const arena = static_cast<char*>(workspace);
    int32_t* acc_base   = reinterpret_cast<int32_t*>(arena + sc.off_acc);
    int8_t* pack_a_base = reinterpret_cast<int8_t*>(arena + sc.off_pack_a);

    const int64_t NG = N * G;
    const int64_t total = NG * oh_chunks;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < total;

    ctx.cpu.run(0, total, [&](int64_t t) {
        const int64_t ng    = t / oh_chunks;
        const int64_t chunk = t % oh_chunks;
        const int64_t n = ng / G;
        const int64_t g = ng % G;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;

        char* const slot_ptr = arena + sc.head + slot * sc.slot_stride;
        TDst* col_data = reinterpret_cast<TDst*>(slot_ptr + sc.off_col);
        int8_t* pack_b = reinterpret_cast<int8_t*>(slot_ptr + sc.off_pack);
        int32_t* acc   = acc_base + slot * plan.acc_size;
        int8_t* pack_a = pack_a_base + slot * plan.pack_a_size;
        int32_t* r_b   = need_col_sum ? r_b_buf.data() + slot * r_b_len : nullptr;

        TOut* output_ptr = out_ptr + n * ob_step + g * og_step;
        const TDst* input_ptr  = in_base + n * ib_step + g * ig_step;
        const int8_t* weight_ptr = w_ptr + g * oc_per_group * w_ocn_step;
        [[maybe_unused]] const TOut* bias_ptr =
            b_ptr ? b_ptr + g * oc_per_group : nullptr;
        [[maybe_unused]] const int32_t* ibias_ptr =
            b_i32 ? b_i32 + g * oc_per_group : nullptr;

        // This chunk's contiguous oh-block range (disjoint across chunks).
        const int64_t b0 = chunk * num_oh_blocks / oh_chunks;
        const int64_t b1 = (chunk + 1) * num_oh_blocks / oh_chunks;

        for (int64_t b = b0; b < b1; ++b) {
            const int64_t oh       = b * oh_block;
            const int64_t ohc      = std::min(oh_block, OH - oh);
            const int64_t roi_area = ohc * OW;

            const int64_t ih = oh * SH - PH;
            const int64_t iw = 0 * SW - PW;
            const int roi_pad_h = (ih < 0) ? static_cast<int>(-ih)
                                           : (ih >= IH ? static_cast<int>(ih - IH) : 0);
            const int roi_pad_w = (iw < 0) ? static_cast<int>(-iw)
                                           : (iw >= IW ? static_cast<int>(iw - IW) : 0);

            // oc is the outer loop: the int32 accumulator holds one oc-block's
            // tile and its partial sums must survive every ic-block. With
            // ic-block outer the next oc-block's fill would wipe them out; for
            // the usual oc_per_group <= ocn_block there is one iteration. Past
            // ocn_block the gather and B pack repeat per oc block, but that is a
            // byte copy by then — quantizing here is what used to be expensive.
            for (int64_t ocn = 0; ocn < oc_per_group; ocn += ocn_block) {
                const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn);

                if (need_col_sum) {
                    std::fill(r_b, r_b + roi_area, 0);
                }

                for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                    const int64_t icnc = std::min(icn_block, ic_per_group - icn);

                    const TDst* input_local = input_ptr + icn * icn_step;
                    // pointwise: B is read straight from the plane instead.
                    if (!plan.pointwise) {
                        // A padded sample is the real value 0, stored as the
                        // activation's zero point.
                        k::tiled_im2col_2d<TDst>(
                            col_data, input_local,
                            0, static_cast<int>(oh),
                            static_cast<int>(OW), static_cast<int>(ohc),
                            roi_pad_h, roi_pad_w,
                            static_cast<int>(PH), static_cast<int>(PW),
                            static_cast<int>(icnc), static_cast<int>(IH), static_cast<int>(IW),
                            static_cast<int>(icn_step), static_cast<int>(in_row_stride),
                            static_cast<int>(KH), static_cast<int>(KW),
                            static_cast<int>(SH), static_cast<int>(SW),
                            static_cast<int>(DH), static_cast<int>(DW),
                            static_cast<TDst>(zp_x));
                    }

                    // B = [K=icnc*karea][N=roi_area]. The u8 shift happens inside
                    // the pack, before the x86 panel sums.
                    const TDst* b_src = plan.pointwise
                        ? input_local + oh * in_row_stride
                        : col_data;
                    const int b_ld = plan.pointwise ? static_cast<int>(icn_step)
                                                    : static_cast<int>(roi_area);
                    const int kdim = static_cast<int>(icnc * karea);

                    tile_pack_rhs_i8(false, static_cast<int>(roi_area), kdim,
                                     pack_b, static_cast<int>(int8_panel_stride(kdim)),
                                     reinterpret_cast<const int8_t*>(b_src), b_ld,
                                     a_biased, /*src_shift=*/kU8);

                    if (need_col_sum) {
                        // Sum the bytes as the packer read them, then move the
                        // unsigned sum into the kernel's shifted space.
                        if constexpr (kU8) {
                            k::int8_col_sum_slab<uint8_t>(b_src, b_ld, 0, roi_area, kdim, r_b);
                            for (int64_t i = 0; i < roi_area; ++i) {
                                r_b[i] -= 128 * kdim;
                            }
                        } else {
                            k::int8_col_sum_slab<int8_t>(b_src, b_ld, 0, roi_area, kdim, r_b);
                        }
                    }

                    // The A panels: in place for a prepacked filter (packed for
                    // exactly this k block and oc block), else rebuilt here.
                    const int8_t* a_panel;
                    if (prepacked) {
                        // From the blob base: pg.offset carries the group.
                        a_panel = w_ptr +
                            pg.offset(g, icn / icn_block, ocn / ocn_block);
                    } else {
                        tile_pack_lhs_i8(false, static_cast<int>(ocnc), kdim,
                                         pack_a, static_cast<int>(geom_ldd_i8_a(kdim)),
                                         weight_ptr + icn * karea + ocn * w_ocn_step,
                                         static_cast<int>(w_ocn_step), a_biased);
                        a_panel = pack_a;
                    }

                    // tile_mma_pack_i8 accumulates, so the first ic-block zeroes
                    // the tile instead of seeding it.
                    if (icn == 0) {
                        std::fill(acc, acc + ocnc * roi_area, 0);
                    }

                    tile_mma_pack_i8(static_cast<int>(ocnc), static_cast<int>(roi_area),
                                     kdim, acc, static_cast<int>(roi_area),
                                     a_panel, pack_b, INT32_MIN, INT32_MAX);
                }

                // The oc-block is complete: rewrite in place while it is hot,
                // dequantized to fp or requantized to the integer destination.
                if constexpr (kIntOut) {
                    // The tile's rows are output channels, so the weight's
                    // per-channel arrays are the per-row ones.
                    Int8EpilogueParams ip{};
                    ip.r_a  = need_comp ? r_a.data() + g * oc_per_group + ocn
                                        : nullptr;
                    ip.s_a  = qw.scale_data ? qw.scale_data + g * oc_per_group + ocn
                                            : nullptr;
                    ip.zp_a = qw.zero_point_data
                            ? qw.zero_point_data + g * oc_per_group + ocn : nullptr;
                    ip.r_b         = need_col_sum ? r_b : nullptr;
                    ip.zp_b        = nullptr;          // per-tensor activation
                    ip.s_a_scalar  = qw.scale;
                    ip.s_b_scalar  = s_x;
                    ip.zp_a_scalar = qw.zero_point;
                    ip.zp_b_scalar = zp_eff;
                    ip.a_bias_offset = a_bias_offset;
                    ip.k    = static_cast<int32_t>(k_full);
                    // Both terms live unless the per-channel half is in bias_fold.
                    ip.compensate = need_col_sum;
                    ip.col_term   = need_col_sum;
                    ip.out_scale = quant_scale_at(qc, 0);
                    ip.zp_out    = quant_zp_at(qc, 0);
                    ip.relu = attrs.epilogue.type == EpilogueActivateType::Relu;
                    ip.bias = !bias_fold.empty()
                            ? bias_fold.data() + g * oc_per_group + ocn
                            : (ibias_ptr ? ibias_ptr + ocn : nullptr);
                    TOut* dst = output_ptr + oh * out_row_stride + ocn * ocn_step;
                    if constexpr (std::is_same_v<TOut, uint8_t>) {
                        ip.out_u8 = dst;
                    } else {
                        ip.out = dst;
                    }
                    ip.ldc_out = static_cast<int>(ocn_step);
                    int8_epilogue_tile(ip, acc, static_cast<int>(roi_area),
                                       0, 0, ocnc, roi_area);
                } else {
                    DqEpilogueParams<TOut> ip{};
                    ip.r_a        = need_comp ? r_a.data() + g * oc_per_group + ocn : nullptr;
                    ip.s_a        = qw.scale_data ? qw.scale_data + g * oc_per_group + ocn
                                                  : nullptr;
                    ip.zp_a       = qw.zero_point_data
                                  ? qw.zero_point_data + g * oc_per_group + ocn : nullptr;
                    ip.r_b        = need_col_sum ? r_b : nullptr;
                    ip.row_off    = row_fold.empty() ? nullptr
                                : row_fold.data() + g * oc_per_group + ocn;
                    ip.zp_b       = nullptr;          // per-tensor activation
                    ip.s_a_scalar = qw.scale;
                    ip.s_b_scalar = s_x;
                    ip.zp_a_scalar = qw.zero_point;
                    ip.zp_b_scalar = zp_eff;
                    ip.a_bias_offset = a_bias_offset;
                    ip.k    = static_cast<int32_t>(k_full);
                    // Both terms unless the row half is in row_off.
                    ip.compensate = need_col_sum;
                    ip.col_term   = need_col_sum;
                    ip.act_u8 = kU8;
                    ip.ep   = &attrs.epilogue;
                    ip.bias = bias_ptr ? bias_ptr + ocn : nullptr;
                    ip.out  = output_ptr + oh * out_row_stride + ocn * ocn_step;
                    ip.ldc_out = static_cast<int>(ocn_step);
                    dq_epilogue_tile<TOut>(ip, acc, static_cast<int>(roi_area),
                                           0, 0, ocnc, roi_area);
                }
            }
        }
    });
}

/// Resolve the integer storage behind an fp activation (`act_quant_dtype`) and
/// run the implementation.
template <class TAct, class TOut>
void conv2d_planar_int8_act(const Conv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            const Conv2DGemmPlan& plan,
                             void* workspace)
{
    constexpr bool kPreQuant =
        std::is_same_v<TAct, int8_t> || std::is_same_v<TAct, uint8_t>;
    if constexpr (kPreQuant) {
        conv2d_planar_int8_impl<TAct, TAct, TOut>(attrs, output, inputs, ctx, plan, workspace);
    } else if (attrs.act_quant_dtype == DataType::u8) {
        conv2d_planar_int8_impl<TAct, uint8_t, TOut>(attrs, output, inputs, ctx, plan, workspace);
    } else {
        conv2d_planar_int8_impl<TAct, int8_t, TOut>(attrs, output, inputs, ctx, plan, workspace);
    }
}

/// Second-level dispatch: the output dtype.
template <class TAct>
void conv2d_planar_int8_out(const Conv2DAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     const Conv2DGemmPlan& plan,
                            void* workspace)
{
    switch (output.data_type()) {
    case DataType::f16:
        conv2d_planar_int8_act<TAct, half>(attrs, output, inputs, ctx, plan, workspace);
        break;
    case DataType::s8:
        conv2d_planar_int8_act<TAct, int8_t>(attrs, output, inputs, ctx, plan, workspace);
        break;
    case DataType::u8:
        conv2d_planar_int8_act<TAct, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
        break;
    default:
        conv2d_planar_int8_act<TAct, float>(attrs, output, inputs, ctx, plan, workspace);
        break;
    }
}

// =========================================================================
//  Plan — single source of truth for block sizes, parallelism + workspace
// =========================================================================

Conv2DGemmPlan make_plan(const Conv2DAttributes& attrs,
                        std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs,
                        int num_threads,
                        bool use_thread_slots,
                        int64_t num_slots)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(outputs.size() >= 1);

    const auto& in  = inputs[0];
    const auto& wt  = inputs[1];
    const auto& out = outputs[0];

    Conv2DGemmPlan plan;
    plan.route = Conv2DGemmRoute::Planar;

    const DataType dt = in.dtype;
    const bool quant = is_quantized_conv2d(wt.dtype);

    if (quant) {
        // Activation: pre-quantized (u8/s8), or fp quantized per call.
        if (dt != DataType::u8 && dt != DataType::s8 &&
            dt != DataType::f32 && dt != DataType::f16) {
            return plan;
        }
        // Output: fp (the activation dtype, or attributes.output_dtype when the
        // activation is already quantized) or an integer s8/u8 requantization.
        const bool int_out = out.dtype == DataType::s8 || out.dtype == DataType::u8;
        if (!int_out && out.dtype != DataType::f32 && out.dtype != DataType::f16) {
            return plan;
        }
        if (!int_out && (dt == DataType::f32 || dt == DataType::f16) &&
            out.dtype != dt) {
            return plan;
        }
        const DataType bias_dt = int_out ? DataType::s32 : out.dtype;
        if (inputs.size() > 2 && inputs[2].dtype != bias_dt) {
            return plan;
        }
        // Only the reference implements add_to's fold.
        if (attrs.add_to) {
            return plan;
        }
    } else {
        if (dt != DataType::f32 && dt != DataType::f16) {
            return plan;
        }
        if (wt.dtype != dt || out.dtype != dt) {
            return plan;
        }
        if (inputs.size() > 2 && inputs[2].dtype != dt) {
            return plan;
        }
    }

    const int64_t N  = in.dims[0];
    const int64_t IC = in.dims[1];
    const int64_t IW = in.dims[3];
    const int64_t OC = wt.dims[0];
    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];
    const int64_t PH    = attrs.padding[0];

    // A 1x1 kernel at unit stride with no padding lets the GEMM read the input
    // plane directly: B = input, ldb = the channel stride, K = ic_per_group.
    // That needs a contiguous output row's worth of input, hence the row-stride
    // test; row_stride_elems == 0 is TensorDesc's "unknown → compact" convention.
    // An fp activation is re-quantized into a compact plane, so its pitch does
    // not disqualify it.
    const bool fp_act = (dt == DataType::f32 || dt == DataType::f16);
    plan.pointwise = (karea == 1 &&
                      attrs.stride[0] == 1 && attrs.stride[1] == 1 &&
                      attrs.padding[0] == 0 && attrs.padding[1] == 0 &&
                      ((quant && fp_act) ||
                       in.row_stride_elems == 0 || in.row_stride_elems == IW));

    // An fp activation becomes a compact plane once per call, then behaves like
    // a pre-quantized one (see the header).
    plan.quant_size = (quant && fp_act) ? N * IC * in.dims[2] * IW : 0;

    // No col buffer bounds the ic-block, so pointwise runs the whole
    // ic_per_group in one pass.
    //
    // The quantized A panels are a load-time artifact (PackedOIHW), so its
    // blocks are conv_packed_weight_planar's; the fp route packs its own B.
    const bool prepacked_a = wt.layout == TensorLayout::PackedOIHW;
    const ConvPackedWeightPlanar pg = quant
        ? conv_packed_weight_planar(attrs, ic_per_group, oc_per_group, wt.dtype)
        : ConvPackedWeightPlanar{};
    plan.icn_block = quant
                   ? pg.icn_block
                   : (plan.pointwise
                          ? ic_per_group
                          : std::min<int64_t>(get_ic_block(static_cast<int>(karea)),
                                              ic_per_group));
    plan.ocn_block = std::min<int64_t>(kOcnBlock, oc_per_group);
    plan.oh_block  = ((PH + 1 + kAlign - 1) / kAlign) * kAlign;

    // The GEMM's N axis is the block's rows flattened into one roi_area-wide
    // run, and every store, seed and epilogue pass addresses it contiguously
    // (ldc is the channel step). A block may therefore only span rows the
    // destination lays out back to back: a pitched output pins it to one row.
    // row_stride_elems == 0 is TensorDesc's unknown → compact convention.
    const bool out_rows_dense =
        out.row_stride_elems == 0 || out.row_stride_elems == OW;
    if (!out_rows_dense) { plan.oh_block = 1; }

    const int64_t NG = N * G;
    const int64_t nt = std::max<int64_t>(num_threads, 1);

    // The oh-block is the atomic unit of work, so the pool is capped at
    // NG·num_oh_blocks tasks. Halve the block while more than half the pool is
    // idle (at 8 threads the same split is a loss: 397→349 GFLOPS), flooring at
    // two n-panels where the weight panel re-read stops paying. col_size shrinks
    // with the block; the expansion loop below grows it back.
    const int64_t nr_max = quant ? NR_MAX_I8
                                 : ((dt == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32);
    for (;;) {
        const int64_t blocks = (OH + plan.oh_block - 1) / plan.oh_block;
        if (2 * NG * blocks >= nt || plan.oh_block <= 1) { break; }
        const int64_t next = plan.oh_block / 2;
        if (next * OW < 2 * nr_max) { break; }
        plan.oh_block = next;
    }

    // Mirror of the halving loop: when the filter rows outnumber the plane they
    // feed, covering the whole plane sweeps the weight once instead of once per
    // block (MLAS's ExpandThenGemm takes the same crossover, convolve.cpp:1691).
    // The block lands on OH itself; the kernel derives the row count per block.
    if (out_rows_dense && oc_per_group > OH * OW) {
        while (plan.oh_block < OH) {
            const int64_t grown = std::min<int64_t>(plan.oh_block * 2, OH);
            const int64_t grown_blocks = (OH + grown - 1) / grown;
            if (2 * NG * grown_blocks < nt) { break; }  // pool still needs the split
            plan.oh_block = grown;
        }
    }

    // Grow the oh-block back while every worker still gets a whole block of its
    // own: the initial block follows the filter, so a 28-row plane runs 4-row
    // blocks that each restart the gather, the packer and their C pass (+10..15%
    // at one thread on the 28-56 row planes, a wash at 112).
    //
    // The block count must stay a multiple of the chunk count (an uneven split
    // cost -12% at 4T on a 112 plane) and must not grow the busiest chunk's rows.
    // The search steps the block count by the chunk count rather than doubling
    // the row count: it can land on 14 rows on a 112 plane, where 4→8→16 cannot.
    if (out_rows_dense && !plan.pointwise) {
        constexpr int64_t kScratchCap = int64_t{256} << 10;   // elements per slot
        const int64_t scratch_per_row = plan.icn_block * karea * OW;
        const int64_t cap_rows = (scratch_per_row > 0) ? kScratchCap / scratch_per_row : OH;
        const int64_t max_rows = std::max<int64_t>(plan.oh_block,
                                                   std::min<int64_t>(OH, cap_rows));
        const int64_t chunks = std::clamp<int64_t>((nt + NG - 1) / NG, 1, OH);
        // Rows the busiest oh-chunk owns under block size `block` (the kernel's
        // own b0..b1 split).
        auto busiest_chunk_rows = [&](int64_t block) {
            const int64_t nblk = (OH + block - 1) / block;
            const int64_t nc = std::min(chunks, nblk);   // oh_chunks <= num_oh_blocks
            int64_t worst = 0;
            for (int64_t c = 0; c < nc; ++c) {
                const int64_t lo = std::min(OH, c * nblk / nc * block);
                const int64_t hi = std::min(OH, (c + 1) * nblk / nc * block);
                worst = std::max(worst, hi - lo);
            }
            return worst;
        };
        const int64_t base_rows = busiest_chunk_rows(plan.oh_block);
        for (int64_t blocks = chunks; blocks <= OH; blocks += chunks) {
            const int64_t grown = (OH + blocks - 1) / blocks;
            if (grown > max_rows) { continue; }          // over the scratch budget
            // ceil(OH/grown) need not hand `blocks` back — only then is the
            // split actually one block per chunk.
            if ((OH + grown - 1) / grown != blocks) { continue; }
            if (busiest_chunk_rows(grown) > base_rows) { continue; }  // long tail
            if (grown > plan.oh_block) { plan.oh_block = grown; }
            break;
        }
    }

    plan.num_oh_blocks = (OH + plan.oh_block - 1) / plan.oh_block;

    // Split oh when (batch, group) alone under-subscribes the pool (batch=1 /
    // groups=1 would otherwise run single-threaded); cap at the oh-block count.
    plan.oh_chunks = std::clamp<int64_t>((nt + NG - 1) / NG, 1, plan.num_oh_blocks);

    // Second parallel axis: the filter rows. A small plane has too few oh-blocks
    // to fill a wide pool (a 7x7 plane is two; a 224 ResNet-18 layer4 lost 155 →
    // 68 GFLOPS going 4T → 20T). Spend this axis first: each oc-chunk reads a
    // *distinct* filter slice, where another oh-chunk re-reads the same rows —
    // only the B gather duplicates, ~1/(2·M) of the GEMM's work (MLAS's
    // ExpandThenGemm, convolve.cpp:1691). Gated like the halving and fp-only.
    // The chunk count is the NEAREST whole number to nt/plane_tasks (see header).
    plan.oc_chunks = 1;
    if (!quant) {
        // Two M-panels per task is the floor, mirroring the oh-block's two
        // N-panels: below it the per-task overheads (bias seed, epilogue, the
        // re-gathered B panel) stop being amortized.
        const int64_t mr = (dt == DataType::f16) ? MR_MAX_F16 : MR_MAX_F32;
        const int64_t plane_tasks = NG * plan.oh_chunks;
        if (2 * plane_tasks < nt) {
            const int64_t max_chunks = std::max<int64_t>(1, oc_per_group / (2 * mr));
            const int64_t want = (nt + plane_tasks / 2) / plane_tasks;
            plan.oc_chunks = std::clamp<int64_t>(want, 1, max_chunks);
        }
    }

    // The fp route fuses gather and pack, so it has no col matrix.
    plan.col_size  = (plan.pointwise || !quant)
                   ? 0
                   : plan.icn_block * karea * plan.oh_block * OW;
    // add_to never reaches here on the quantized route (gated above).
    plan.orig_size = (attrs.add_to && !quant)
                   ? plan.ocn_block * plan.oh_block * OW : 0;

    // Packed-B scratch at the widest n-block and longest k-block; the packer's
    // stride comes from the actual k, so this is a bound, not a worst case.
    // The quantized route's is in bytes (esz 1) with the int8 panel stride.
    const int64_t esz      = quant ? 1 : static_cast<int64_t>(data_type_size(dt));
    const int64_t kdim_max = plan.icn_block * karea;
    plan.pack_size = ((plan.oh_block * OW + nr_max - 1) / nr_max)
                   * gemm_panel_slot(kdim_max, nr_max, esz, quant);

    // One slot per worker when the backend reports ids and there are fewer of
    // them than tasks; else one per task.
    const int64_t total = NG * plan.oh_chunks * plan.oc_chunks;
    const int64_t slots = (num_slots > 0) ? num_slots : nt;
    plan.num_slots = (use_thread_slots && slots < total) ? slots : total;

    if (quant) {
        // Both regions sit behind the slots in the shared arena.
        plan.acc_size = plan.ocn_block * plan.oh_block * OW;
        // A prepacked filter's panels are read in place: no per-slot pack region.
        plan.pack_a_size = prepacked_a
            ? 0
            : static_cast<int64_t>(
                  num_panels_max(static_cast<int>(plan.ocn_block), MR_I8))
              * geom_ldd_i8_a(kdim_max);
    }
    plan.workspace_size = static_cast<size_t>(gemm_scratch_layout(plan, esz).total);
    return plan;
}


// ---- dispatch -------------------------------------------------------------

/// Resolve the activation's own type; the plan already validated the combination.
Status conv2d_planar_int8_run(const Conv2DAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx, const Conv2DGemmPlan& plan,
                                void* workspace)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: conv2d_planar_int8_out<float>(attrs, output, inputs, ctx, plan, workspace); break;
    case DataType::f16: conv2d_planar_int8_out<half>(attrs, output, inputs, ctx, plan, workspace); break;
    case DataType::u8:  conv2d_planar_int8_out<uint8_t>(attrs, output, inputs, ctx, plan, workspace); break;
    case DataType::s8:  conv2d_planar_int8_out<int8_t>(attrs, output, inputs, ctx, plan, workspace); break;
    default: return Status::unsupported;
    }
    return Status::ok;
}

/// An s8 weight selects the quantized driver; otherwise the two fp element types.
Status run(const Conv2DAttributes& attrs, TensorView& output,
           std::span<const TensorView> inputs, const ComputeContext& ctx,
           const Conv2DGemmPlan& plan, void* workspace)
{
    if (is_quantized_conv2d(inputs[1].data_type())) {
        return conv2d_planar_int8_run(attrs, output, inputs, ctx, plan, workspace);
    }
    switch (inputs[0].data_type()) {
    case DataType::f32:
        conv2d_planar_impl<float>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    case DataType::f16:
        conv2d_planar_impl<half>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

}  // anonymous namespace
}  // namespace planar

// =========================================================================
//  Channel-last route (NHWC) — pixels on M, the packed HWIO weight on N
// =========================================================================

namespace channel_last {
namespace {

using nnops::simd::s_load;
using nnops::simd::s_store;

// ---- block-size heuristics ------------------------------------------------

/// Input-channel block: kdim = ic_block × karea lands in ~144–256, and the
/// packed-B panel (nr_max × kdim) stays near L1/L2. A 1x1 filter has neither a
/// gather nor a karea multiplier, so K runs four times as wide.
inline int64_t get_ic_block_nhwc(int64_t karea, int64_t ic_per_group) noexcept {
    if (karea == 1)  { return 256; }
    // A narrow channel count takes itself whole even when the ladder would split
    // it: with every channel of a pixel in the block, one filter row's taps are
    // adjacent, so the gather copies one run instead of `KW` runs of two. That
    // is the 3-channel stem's whole loss (IC=3, 7x7).
    if (ic_per_group * karea <= 256) { return std::max<int64_t>(ic_per_group, 1); }
    if (karea <= 9)  { return 16; }
    if (karea <= 25) { return 8;  }
    if (karea <= 49) { return 2;  }
    return 1;
}

constexpr int64_t kOcnBlock = 128;  ///< output-channel tile (the GEMM's N)
/// An oc_per_group at or below this stays one block: every extra oc block costs
/// a second pass over the pixels (A pack, accumulator fill, epilogue). Splitting
/// 144 into 80 + 64 cost the int8 expand layer a quarter of its time.
constexpr int64_t kOcnSingleCap = 192;
constexpr int64_t kMBlock   = 24;   ///< pixel tile (the GEMM's M): 4 × mr_max(6)
/// Pixels a task needs before it is worth splitting the plane again: two M
/// blocks, i.e. one full tile plus a second to hide its prologue.
constexpr int64_t kTaskMinPixels = 2 * kMBlock;
/// Bytes one accumulator pass may cover (the plan's rows-per-pass knob).
constexpr int64_t kPassAccBytes = 128 * 1024;
constexpr int64_t kABSlack  = 8;    ///< elements past the A tile (vector over-read)

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
void tile_seed(T* NNOPS_RESTRICT tile, int ld, int mc, int nc,
               const T* NNOPS_RESTRICT bias, bool add) noexcept
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
template <class T>
void tile_zero(T* NNOPS_RESTRICT tile, int ld, int mc, int nc) noexcept
{
    for (int m = 0; m < mc; ++m) {
        T* row = tile + static_cast<int64_t>(m) * ld;
        for (int i = 0; i < nc; ++i) { row[i] = T(0); }
    }
}

/// Copy a strided tile (row stride ld) into a dense [M][N] scratch.
template <class T>
void tile_save(T* NNOPS_RESTRICT dst, const T* NNOPS_RESTRICT src,
               int ld, int M, int N) noexcept
{
    for (int m = 0; m < M; ++m) {
        const T* s = src + static_cast<int64_t>(m) * ld;
        T* d = dst + static_cast<int64_t>(m) * N;
        for (int i = 0; i < N; ++i) { d[i] = s[i]; }
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
void pack_block(T* dst, int64_t panel_slot, T* stage,
                const T* w, int64_t w_ic_stride, int64_t w_tap_stride,
                int64_t ocnc, int64_t icnc,
                int karea, int64_t nr_max, int kdim) noexcept
{
    (void)nr_max;
    if (karea == 1) {
        tile_pack_rhs(false, static_cast<int>(ocnc), static_cast<int>(icnc),
                      dst, static_cast<int>(panel_slot),
                      w, static_cast<int>(w_ic_stride), 1.0f);
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
        tile_pack_rhs(false, static_cast<int>(np), kdim,
                      dst + (p0 / nr_max) * panel_slot, static_cast<int>(panel_slot),
                      stage, static_cast<int>(nr_max), 1.0f);
    }
}

// ---- main kernel ----------------------------------------------------------

template <class T>
void conv2d_channel_last_impl(const Conv2DAttributes& attrs,
                              TensorView& output,
                              std::span<const TensorView> inputs,
                              const ComputeContext& ctx,
                              const Conv2DGemmPlan& plan,
                              void* workspace)
{
    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Logical [N, C, H, W]; NHWC stores [N][H][W][C], so a "row" is one spatial
    // line and a pixel's channel vector is contiguous.
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // HWIO: the output channel is the last axis; a prepacked blob leads with it.
    const bool prepacked = is_packed_conv_weight(weight.layout());
    const int64_t OC = prepacked ? weight.shape(0) : weight.shape(3);

    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

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
    const int      m_block  = static_cast<int>(plan.m_block);
    const int64_t oh_chunks = plan.oh_chunks;
    const int64_t ph_block  = plan.ph_block;
    const int64_t oc_chunks = plan.oc_chunks;

    // A sample is IH rows of the channel-last pitch.
    const int64_t ib_step = IH * in_row_stride;
    const int64_t ob_step = OH * out_row_stride;

    // HWIO (TensorFlow's filter): [KH][KW][IC/G][OC]. The output channel is the
    // contiguous axis, so one (tap, input channel) is an OC-length run and the B
    // panel is a plain strided copy — no transpose. Dims, not pitch: a weight is
    // a load-time constant. A prepacked blob is read only through the panel
    // geometry, so its strides stay 0.
    //   w_ic_stride = OC (one input channel's row); w_tap_stride = ic_per_group·OC
    const int64_t w_row = prepacked ? 0 : weight.shape(3);   // [oc] run of one (tap, ic)
    const int64_t w_ic_stride = w_row;                       // [ic] within a tap
    const int64_t w_tap_stride = prepacked                   // [kh][kw] position
        ? 0 : weight.shape(2) * w_row;

    T* out_ptr = output.ptr<T>();
    const T* in_ptr = input.ptr<T>();
    const T* w_ptr  = weight.ptr<T>();
    const T* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    constexpr int64_t nr_max = nr_max_flt<T>();
    const int64_t esz = static_cast<int64_t>(sizeof(T));
    constexpr float kInf = std::numeric_limits<float>::infinity();

    // ---- the finish decision ----------------------------------------------
    // A pure clamp folds into the MMA store, as on the planar route. The bias is
    // per *column* here (rows are pixels), so it seeds the tile instead of the
    // micro-kernel's per-row seed — leaving the clamp nothing else to do.
    float clamp_lo = -kInf;
    float clamp_hi = kInf;
    bool epi_folded = false;
    if (!attrs.add_to) {
        switch (attrs.epilogue.type) {
        case EpilogueActivateType::Relu:
            clamp_lo = 0.0f; epi_folded = true;
            break;
        case EpilogueActivateType::Relu6:
            clamp_lo = 0.0f; clamp_hi = 6.0f; epi_folded = true;
            break;
        default:
            break;
        }
    }
    const bool epi_active = attrs.epilogue.type != EpilogueActivateType::None;
    const bool epi_pass   = epi_active && !epi_folded;

    // add_to folds the pre-existing output into the accumulator seed when the
    // epilogue is the identity; with an activation the residual must be added
    // *after* it, so the original tile is preserved instead.
    const bool keep_orig = attrs.add_to && epi_pass;

    // The bias is per column, so it always seeds the tile; add_to's seed also
    // carries the residual. A seeded tile must never be zeroed by the MMA.
    const bool seed_bias = attrs.add_to && has_bias;
    const bool bias_seed = has_bias && !attrs.add_to;
    const bool need_epi  = epi_pass;

    // ---- block geometry ----------------------------------------------------
    // Same function prepackWeights built the blob with; it must never disagree.
    const ConvPackedWeightLast pg =
        conv_packed_weight_last(attrs, ic_per_group, oc_per_group, weight.data_type());
    NNOPS_ASSERT(pg.icn_block == icn_block && pg.ocn_block == ocn_block &&
                 pg.nr == nr_max);
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
    const T* const panels = prepacked ? w_ptr : (shared ? shared_ptr : nullptr);

    // ---- the shared weight prepack (once per call) -------------------------
    // The same entry prepackWeights uses, only the destination differs.
    if (shared) {
        const int64_t cols = (plan.shared_elems + OC - 1) / OC;
        const int64_t shape[] = {OC, cols};
        TensorView blob(std::span<const int64_t>(shape, 2), weight.data_type(),
                        shared_ptr, cols * esz, TensorLayout::PackedHWIO);
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
    std::vector<T> bias_pad;
    if (bias_seed) {
        bias_pad.assign(static_cast<size_t>(OC + nr_max), T{});
        std::memcpy(bias_pad.data(), b_ptr, static_cast<size_t>(OC) * sizeof(T));
    }

    const auto run_task = [&](int64_t t) {
        const int64_t ng   = t / chunks_per_ng;
        const int64_t rem  = t % chunks_per_ng;
        // The oc-chunk varies fastest: neighbours in task order then cover
        // different filter rows, so their weight streams differ.
        const int64_t chunk    = rem / oc_chunks;
        const int64_t oc_chunk = rem % oc_chunks;
        const int64_t n = ng / G;
        const int64_t g = ng % G;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;
        char* const slot_ptr = base + sc.head + slot * sc.slot_stride;
        T* const a_buf    = reinterpret_cast<T*>(slot_ptr + sc.off_a);
        T* const orig_buf = reinterpret_cast<T*>(slot_ptr + sc.off_orig);
        T* const stage    = reinterpret_cast<T*>(slot_ptr + sc.off_stage);
        T* const pack_buf = reinterpret_cast<T*>(slot_ptr + sc.off_pack);

        T* const out_g = out_ptr + n * ob_step + g * oc_per_group;
        const T* const in_g = in_ptr + n * ib_step + g * ic_per_group;
        const T* const w_g  = w_ptr + g * oc_per_group;
        const T* const bias_g = b_ptr ? b_ptr + g * oc_per_group : nullptr;
        const T* const bias_col = bias_seed ? bias_pad.data() + g * oc_per_group : nullptr;

        // This oh-chunk's contiguous row range, and its oc range. The oc split
        // is over *panels*, not channels: the blob holds panel p as channels
        // [p·nr, (p+1)·nr) of its k column, so a task starts where a panel does.
        const int64_t r0 = chunk * OH / oh_chunks;
        const int64_t r1 = (chunk + 1) * OH / oh_chunks;
        const int64_t oc_lo = oc_chunk * pg.panels_oc / oc_chunks * nr_max;
        const int64_t oc_hi = std::min(
            (oc_chunk + 1) * pg.panels_oc / oc_chunks * nr_max, oc_per_group);

        // Dense rows make a pixel tile one contiguous run, so it may cross a row
        // boundary — which keeps a narrow plane's tiles at the full M instead of
        // paying MR-row granularity per row (a 7-pixel row of a 6-row MR block
        // wastes 40%). Pointwise reads A the same way, so the input must be dense.
        const bool flat = out_row_stride == OW * OC &&
                          (!plan.pointwise || in_row_stride == IW * IC);

        for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block) {
            const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);
            // The residual region covers one pass's rows (one oc block live).
            T* const orig = keep_orig ? orig_buf : nullptr;

            for (int64_t pr = r0; pr < r1; pr += ph_block) {
            const int64_t pr1 = std::min(r1, pr + ph_block);
            const int64_t p_first = pr * OW;
            const int64_t p_last  = pr1 * OW;
            // A pass of 49 pixels is 24 + 24 + 1 and the last tile pays a whole
            // MR block for one row. Re-split only a *short* tail, and keep the
            // step a multiple of MR: 24 + 24 + 12 costs the same MR blocks as
            // three even 16s, where a step of 23 would pay a fourth block for 5.
            constexpr int64_t mr_step =
                std::is_same_v<T, float> ? MR_MAX_F32 : MR_MAX_F16;
            const int64_t p_span = std::max<int64_t>(p_last - p_first, 1);
            const int64_t p_tail = p_span % m_block;
            const int64_t p_step =
                (flat && p_tail != 0 && p_tail < mr_step && m_block >= 2 * mr_step)
                    ? (m_block / mr_step - 1) * mr_step
                    : m_block;

            for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                const bool is_last_icn = (icn + icn_block) >= ic_per_group;
                const int kdim = static_cast<int>(icnc * karea);

                const T* b_panel;
                if (panels) {
                    // The k block's panel column, then this task's own panels
                    // along it: no pack, however many chunks the oc axis took.
                    const int64_t ib = icn / icn_block;
                    b_panel = panels + pg.offset(g, ib) +
                              (ocn / nr_max) * pg.slot(ib);
                } else {
                    pack_block<T>(pack_buf, gemm_panel_slot(kdim, nr_max, esz, false),
                                  stage, w_g + ocn + icn * w_ic_stride,
                                  w_ic_stride, w_tap_stride, ocnc, icnc, karea,
                                  nr_max, kdim);
                    b_panel = pack_buf;
                }

                const bool clamp_here = is_last_icn && epi_folded;
                const float c_lo = clamp_here ? clamp_lo : -kInf;
                const float c_hi = clamp_here ? clamp_hi : kInf;

                for (int64_t p = p_first; p < p_last; ) {
                    const int64_t oh = p / OW;
                    const int64_t ow0 = p - oh * OW;
                    const int64_t left = flat ? std::min(p_step, p_last - p)
                                              : (OW - ow0);
                    const int mc = static_cast<int>(std::min<int64_t>(m_block, left));

                    const T* a;
                    int lda;
                    if (plan.pointwise) {
                        // The pixel is its own patch: A is the input run itself
                        // (contiguous across a row boundary only when flat).
                        a = in_g + oh * in_row_stride + ow0 * IC + icn;
                        lda = static_cast<int>(IC);
                    } else {
                        tiled_im2col_nhwc<T>(a_buf, kdim, in_g + icn, in_row_stride, IC,
                                       T(0), p, static_cast<int>(OW), flat, mc,
                                       static_cast<int>(IH), static_cast<int>(IW),
                                       KH, KW, SH, SW, DH, DW, PH, PW,
                                       static_cast<int>(icnc), karea);
                        a = a_buf;
                        lda = kdim;
                    }

                    T* const tile = out_g + oh * out_row_stride + ow0 * OC + ocn;
                    T* const orig_tile = orig
                        ? orig + (p - p_first) * ocn_block
                        : nullptr;

                        if (icn == 0 && attrs.add_to) {
                            if (keep_orig) {
                                tile_save<T>(orig_tile, tile, static_cast<int>(OC), mc,
                                             static_cast<int>(ocnc));
                            }
                            if (seed_bias) {
                                // The residual stays in the accumulator only on the
                                // identity-epilogue route; keep_orig saves it and
                                // the tile is the seed.
                                tile_seed<T>(tile, static_cast<int>(OC), mc,
                                             static_cast<int>(ocnc), bias_g + ocn,
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
                        tile_mma_direct(mc, static_cast<int>(ocnc), kdim,
                                        tile, static_cast<int>(OC),
                                        a, lda, b_panel, c_lo, c_hi,
                                        /*zero_mode=*/icn == 0 && !attrs.add_to &&
                                                       !bias_seed,
                                        nullptr,
                                        (icn == 0 && bias_seed) ? bias_col + ocn
                                                                : nullptr);

                        if (is_last_icn && need_epi) {
                            epilogue_inplace<T>(mc, static_cast<int>(ocnc), tile,
                                                static_cast<int>(OC),
                                                nullptr,
                                                attrs.epilogue,
                                                keep_orig ? orig_tile : nullptr,
                                                keep_orig);
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
// per call, as on the planar route.

namespace k = nnops::kernel;  // pure SIMD kernels: the quantizer, int8 reductions

constexpr int64_t align4(int64_t v) noexcept { return (v + 3) & ~int64_t{3}; }

/// True when any zero point the tensor carries is non-zero.
inline bool any_zp_nonzero(const QuantParams& q, int64_t n) noexcept {
    if (q.zero_point_data == nullptr) { return q.zero_point != 0; }
    for (int64_t i = 0; i < n; ++i) {
        if (q.zero_point_data[i] != 0) { return true; }
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
    constexpr bool kPreQuant =
        std::is_same_v<TAct, int8_t> || std::is_same_v<TAct, uint8_t>;

    const int64_t IC = in.shape(1);
    const int64_t IH = in.shape(2);
    const int64_t IW = in.shape(3);
    const int64_t N  = in.shape(0);
    const int64_t rs = in.row_stride_elems();
    const TAct* src  = in.ptr<TAct>();

    const float s = scale;
    const float z = static_cast<float>(zero);

    const int64_t row_len = IW * IC;
    const int64_t rows = N * IH;
    const int64_t total = rows * row_len;
    if (total == 0) { return; }

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
                k::quant_kernel::quantization<int8_t>(
                    static_cast<int>(1), static_cast<int>(len), d, 1, p, 1, &s, &z);
            }
            e += len;
        }
    };

    // A dispatch costs more than the pass: same split as the planar quantizer.
    constexpr int64_t kInlineMax = int64_t{1} << 19;

    const int64_t nt = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    if (nt == 1 || total <= kInlineMax) {
        plane_range(0, total);
        return;
    }
    const int64_t slab = (total + nt - 1) / nt;
    const int64_t nslab = (total + slab - 1) / slab;
    ctx.cpu.run(0, nslab, [&](int64_t i) {
        plane_range(i * slab, std::min(i * slab + slab, total));
    });
}

// ---- the packed-B route ---------------------------------------------------

/// pack_block's integer twin: same panel, same tap-major restaging.
///
/// @p a_biased must be set whenever the kernel folds an A-side bias out of its
/// accumulators through the panel's column sums — on x86 always, since that is
/// where the +128 the VNNI kernel adds to every A group is cancelled.
inline void pack_block_i8(int8_t* dst, int64_t panel_slot, int8_t* stage,
                          const int8_t* w, int64_t w_ic_stride, int64_t w_tap_stride,
                          int64_t ocnc, int64_t icnc,
                          int karea, int64_t nr_max, int kdim, bool a_biased) noexcept
{
    if (karea == 1) {
        tile_pack_rhs_i8(false, static_cast<int>(ocnc), static_cast<int>(icnc),
                         dst, static_cast<int>(panel_slot), w,
                         static_cast<int>(w_ic_stride), a_biased, false);
        return;
    }

    for (int64_t p0 = 0; p0 < ocnc; p0 += nr_max) {
        const int64_t np = std::min(nr_max, ocnc - p0);
        for (int t = 0; t < karea; ++t) {
            for (int64_t i = 0; i < icnc; ++i) {
                std::memcpy(stage + (static_cast<int64_t>(t) * icnc + i) * nr_max,
                            w + t * w_tap_stride + i * w_ic_stride + p0,
                            static_cast<size_t>(np));
            }
        }
        tile_pack_rhs_i8(false, static_cast<int>(np), kdim,
                         dst + (p0 / nr_max) * panel_slot, static_cast<int>(panel_slot),
                         stage, static_cast<int>(nr_max), a_biased, false);
    }
}

// ---- main kernel ----------------------------------------------------------

template <class TAct, class TOut>
void conv2d_channel_last_int8_impl(const Conv2DAttributes& attrs,
                                   TensorView& output,
                                   std::span<const TensorView> inputs,
                                   const ComputeContext& ctx,
                                   const Conv2DGemmPlan& plan,
                                   void* workspace)
{
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    constexpr bool kU8 = std::is_same_v<TAct, uint8_t>;
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
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

    const int KH = static_cast<int>(attrs.kernel_size[0]);
    const int KW = static_cast<int>(attrs.kernel_size[1]);
    const int SH = static_cast<int>(attrs.stride[0]);
    const int SW = static_cast<int>(attrs.stride[1]);
    const int DH = static_cast<int>(attrs.dilation[0]);
    const int DW = static_cast<int>(attrs.dilation[1]);
    const int PH = static_cast<int>(attrs.padding[0]);
    const int PW = static_cast<int>(attrs.padding[1]);
    const int karea = KH * KW;
    const int64_t k_full = ic_per_group * karea;   ///< the epilogue's K

    const int64_t icn_block = plan.icn_block;
    const int64_t ocn_block = plan.ocn_block;
    const int      m_block  = static_cast<int>(plan.m_block);
    const int64_t oh_chunks = plan.oh_chunks;
    const int64_t ph_block  = plan.ph_block;
    const int64_t oc_chunks = plan.oc_chunks;

    const int64_t ob_step = OH * out_row_stride;

    // HWIO: dims, not pitch — one (tap, input channel) is one OC-length run.
    const int64_t w_row = prepacked ? 0 : weight.shape(3);
    const int64_t w_ic_stride = w_row;
    const int64_t w_tap_stride = prepacked ? 0 : weight.shape(2) * w_row;

    // Same geometry prepackWeights used (see the float kernel).
    const ConvPackedWeightLast pg =
        conv_packed_weight_last(attrs, ic_per_group, oc_per_group, weight.data_type());
    NNOPS_ASSERT(pg.icn_block == icn_block && pg.ocn_block == ocn_block &&
                 pg.nr == NR_MAX_I8);
    constexpr int64_t nr_max = NR_MAX_I8;

    // ---- quantization parameters -----------------------------------------
    const QuantParams& qa = input.quant_params();
    const QuantParams& qw = weight.quant_params();

    const float   s_x  = quant_scale_at(qa, 0);
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
    const bool comp     = any_zp_nonzero(qw, OC);

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
    [[maybe_unused]] const TOut* b_ptr =
        (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
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
        prepacked ? w_ptr : (shared ? shared_ptr : nullptr);

    // ---- the shared weight prepack (once per call) -------------------------
    // Through the entry prepackWeights uses, so the addressing is one code path.
    // It runs before the sums are read back below.
    if (shared) {
        const int64_t cols = (plan.shared_elems + OC - 1) / OC;
        const int64_t shape[] = {OC, cols};
        TensorView blob(std::span<const int64_t>(shape, 2), weight.data_type(),
                        shared_ptr, cols, TensorLayout::PackedHWIO);
        conv_packed_weight_pack(attrs, weight, blob, ctx);
    }

    // R_b[oc] = Σ_k weight[oc,k] over the whole K. A packed blob carries it, so
    // the walk below is only for a filter too large for the arena. Serial, for
    // the reason the planar route gives.
    //
    // In HWIO one output channel's K values are a *strided* column, so the sum
    // accumulates over the k rows — elementwise across OC, the contiguous axis.
    std::vector<int32_t> r_b_all;
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

    const int64_t NG = N * G;
    const int64_t chunks_per_ng = oh_chunks * oc_chunks;
    const int64_t total = NG * chunks_per_ng;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < total;

    const auto run_task = [&](int64_t t) {
        const int64_t ng   = t / chunks_per_ng;
        const int64_t rem  = t % chunks_per_ng;
        const int64_t chunk    = rem / oc_chunks;
        const int64_t oc_chunk = rem % oc_chunks;
        const int64_t n = ng / G;
        const int64_t g = ng % G;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;
        char* const slot_ptr = base + sc.head + slot * sc.slot_stride;
        int8_t* const a_pack   = reinterpret_cast<int8_t*>(slot_ptr + sc.off_a);
        int8_t* const a_buf    = a_pack + plan.a_pack_stride;
        int8_t* const stage    = reinterpret_cast<int8_t*>(slot_ptr + sc.off_stage);
        int8_t* const pack_buf = reinterpret_cast<int8_t*>(slot_ptr + sc.off_pack);
        int32_t* const acc_buf = reinterpret_cast<int32_t*>(
            base + sc.off_acc + slot * plan.acc_size * static_cast<int64_t>(sizeof(int32_t)));

        TOut* const out_g = out_ptr + n * ob_step + g * oc_per_group;
        const uint8_t* const in_g = in_base + n * ib_step + g * ic_per_group;
        const int8_t* const w_g  = w_ptr + g * oc_per_group;
        [[maybe_unused]] const TOut* const bias_g = b_ptr ? b_ptr + g * oc_per_group : nullptr;
        [[maybe_unused]] const int32_t* const ibias_g =
            b_i32 ? b_i32 + g * oc_per_group : nullptr;

        const int64_t r0 = chunk * OH / oh_chunks;
        const int64_t r1 = (chunk + 1) * OH / oh_chunks;
        // Over *panels*, like the fp route: a task starts where a panel does.
        const int64_t oc_lo = oc_chunk * pg.panels_oc / oc_chunks * nr_max;
        const int64_t oc_hi = std::min(
            (oc_chunk + 1) * pg.panels_oc / oc_chunks * nr_max, oc_per_group);

        // Same flat-pixel rule as the fp route; pointwise also needs the input
        // dense, since it reads A straight out of the row.
        const bool flat = out_row_stride == OW * OC &&
                          (!plan.pointwise || in_row_stride == IW * IC);

        for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block) {
            const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);

            for (int64_t pr = r0; pr < r1; pr += ph_block) {
            const int64_t pr1 = std::min(r1, pr + ph_block);
            const int64_t p_first = pr * OW;
            const int64_t p_last  = pr1 * OW;
            // Re-split a tail shorter than MR evenly, keeping the step a
            // multiple of MR so the re-split costs no extra block.
            constexpr int64_t mr_step = MR_MAX_I8;
            const int64_t p_span = std::max<int64_t>(p_last - p_first, 1);
            const int64_t p_tail = p_span % m_block;
            const int64_t p_step =
                (flat && p_tail != 0 && p_tail < mr_step && m_block >= 2 * mr_step)
                    ? (m_block / mr_step - 1) * mr_step
                    : m_block;

            for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                const bool is_last_icn = (icn + icn_block) >= ic_per_group;
                const int kdim = static_cast<int>(icnc * karea);
                const int kdim4 = static_cast<int>(align4(kdim));
                // Pointwise reads the pixels straight out of the row, which needs
                // a whole number of 4-byte k groups; else the gather zero-fills.
                const bool direct_a = plan.pointwise && (kdim == kdim4);

                const int8_t* b_panel;
                if (panels) {
                    const int64_t ib = icn / icn_block;
                    b_panel = panels + pg.offset(g, ib) +
                              (ocn / nr_max) * pg.slot(ib);
                } else {
                    pack_block_i8(pack_buf, gemm_panel_slot(kdim, nr_max, 1, true), stage,
                                  w_g + ocn + icn * w_ic_stride,
                                  w_ic_stride, w_tap_stride, ocnc, icnc, karea, nr_max,
                                  kdim, a_biased);
                    b_panel = pack_buf;
                }

                // `mc` lives outside the loop: it is the step, and the body's
                // early exits still have to advance it.
                int mc = 0;
                for (int64_t p = p_first; p < p_last; p += flat ? p_step : mc) {
                    const int64_t oh = p / OW;
                    const int64_t ow0 = p - oh * OW;
                    const int64_t left = flat ? std::min(p_step, p_last - p)
                                              : (OW - ow0);
                    mc = static_cast<int>(std::min<int64_t>(m_block, left));
                    {
                        const uint8_t* const in_row = in_g + oh * in_row_stride;

                        const int8_t* a;
                        int lda;
                        if (direct_a) {
                            // The pixel is its own patch: A is the input run.
                            a = reinterpret_cast<const int8_t*>(
                                in_row + ow0 * IC + icn);
                            lda = static_cast<int>(IC);
                        } else {
                            tiled_im2col_nhwc_i8(a_buf, kdim4, in_g + icn,
                                           in_row_stride, IC, xr,
                                           static_cast<uint8_t>(plane ? zp_x + d : zp_x),
                                           p, static_cast<int>(OW), flat, mc,
                                           static_cast<int>(IH), static_cast<int>(IW),
                                           KH, KW, SH, SW, DH, DW, PH, PW,
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
                        const int64_t ldd_a = geom_ldd_i8_a(kdim4);
                        tile_pack_lhs_i8(false, mc, kdim4, a_pack,
                                         static_cast<int>(ldd_a), a, lda, a_biased);
                        if (icn == 0) {
                            tile_mma_pack_i8_zero(mc, static_cast<int>(ocnc), kdim4,
                                                  acc_tile, static_cast<int>(ocn_block),
                                                  a_pack, b_panel, INT32_MIN, INT32_MAX);
                        } else {
                            tile_mma_pack_i8(mc, static_cast<int>(ocnc), kdim4,
                                             acc_tile, static_cast<int>(ocn_block),
                                             a_pack, b_panel, INT32_MIN, INT32_MAX);
                        }

                        if (!is_last_icn) { continue; }

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
                                for (int m = 0; m < mc; ++m) { ra_tile[m] = 0; }
                                for (int64_t icn2 = 0; icn2 < ic_per_group; icn2 += icn_block) {
                                    const int64_t icnc2 = std::min(icn_block, ic_per_group - icn2);
                                    const int kd2 = static_cast<int>(icnc2 * karea);
                                    const int kd24 = static_cast<int>(align4(kd2));
                                    tiled_im2col_nhwc_i8(a_buf, kd24, in_g + icn2,
                                                   in_row_stride, IC, xr,
                                                   static_cast<uint8_t>(plane ? zp_x + d : zp_x),
                                                   p, static_cast<int>(OW), flat, mc,
                                                   static_cast<int>(IH), static_cast<int>(IW),
                                                   KH, KW, SH, SW, DH, DW, PH, PW,
                                                   static_cast<int>(icnc2), karea, kd2, kd24);
                                    for (int m = 0; m < mc; ++m) {
                                        ra_tile[m] += k::int8_row_sum<int8_t>(
                                            a_buf + static_cast<int64_t>(m) * kd24, kd2);
                                    }
                                }
                            }
                            const int32_t shift_total = d * static_cast<int32_t>(k_full);
                            for (int m = 0; m < mc; ++m) { ra_tile[m] -= shift_total; }
                            ra_ptr = ra_tile;
                        }

                        TOut* const dst = out_g + oh * out_row_stride + ow0 * OC + ocn;
                        if constexpr (kIntOut) {
                            Int8EpilogueParams ip{};
                            ip.r_a  = ra_ptr;
                            ip.s_a  = nullptr;              // per-tensor activation
                            ip.zp_a = nullptr;
                            ip.r_b  = col_term
                                    ? r_b_all.data() + g * oc_per_group + ocn : nullptr;
                            ip.s_b  = qw.scale_data
                                    ? qw.scale_data + g * oc_per_group + ocn : nullptr;
                            ip.zp_b = qw.zero_point_data
                                    ? qw.zero_point_data + g * oc_per_group + ocn : nullptr;
                            ip.s_a_scalar    = s_x;
                            ip.s_b_scalar    = qw.scale;
                            ip.zp_a_scalar   = zp_x;
                            ip.zp_b_scalar   = qw.zero_point;
                            ip.a_bias_offset = d;
                            ip.k    = static_cast<int32_t>(k_full);
                            ip.relu = attrs.epilogue.type == EpilogueActivateType::Relu;
                            ip.compensate = comp;
                            ip.col_term   = col_term;
                            ip.out_scale  = quant_scale_at(qc, 0);
                            ip.zp_out     = quant_zp_at(qc, 0);
                            ip.bias     = nullptr;          // the bias is per column
                            ip.col_bias = ibias_g ? ibias_g + ocn : nullptr;
                            if constexpr (std::is_same_v<TOut, uint8_t>) {
                                ip.out_u8 = dst;
                            } else {
                                ip.out = dst;
                            }
                            ip.ldc_out = static_cast<int>(OC);
                            int8_epilogue_tile(ip, acc_tile, static_cast<int>(ocn_block),
                                               0, 0, mc, ocnc);
                        } else {
                            DqEpilogueParams<TOut> ip{};
                            ip.r_a  = ra_ptr;
                            ip.s_a  = nullptr;
                            ip.zp_a = nullptr;
                            ip.r_b  = col_term
                                    ? r_b_all.data() + g * oc_per_group + ocn : nullptr;
                            ip.s_b  = qw.scale_data
                                    ? qw.scale_data + g * oc_per_group + ocn : nullptr;
                            ip.zp_b = qw.zero_point_data
                                    ? qw.zero_point_data + g * oc_per_group + ocn : nullptr;
                            ip.row_off       = nullptr;
                            ip.s_a_scalar    = s_x;
                            ip.s_b_scalar    = qw.scale;
                            ip.zp_a_scalar   = zp_x;
                            ip.zp_b_scalar   = qw.zero_point;
                            ip.a_bias_offset = d;
                            ip.k             = static_cast<int32_t>(k_full);
                            ip.compensate    = comp;
                            ip.col_term      = col_term;
                            ip.act_u8        = u8a;
                            ip.ep            = &attrs.epilogue;
                            ip.bias          = nullptr;
                            ip.col_bias      = bias_g ? bias_g + ocn : nullptr;
                            ip.out           = dst;
                            ip.ldc_out       = static_cast<int>(OC);
                            dq_epilogue_tile<TOut>(ip, acc_tile,
                                                   static_cast<int>(ocn_block),
                                                   0, 0, mc, ocnc);
                        }
                    }
                }
            }
            }
        }
    };

    ctx.cpu.run(0, total, run_task);
}

// ---- dispatch -------------------------------------------------------------

template <class TAct>
void conv2d_channel_last_int8_out(const Conv2DAttributes& attrs, TensorView& output,
                                  std::span<const TensorView> inputs, const ComputeContext& ctx,
                                  const Conv2DGemmPlan& plan, void* workspace)
{
    switch (output.data_type()) {
    case DataType::f16:
        conv2d_channel_last_int8_impl<TAct, half>(attrs, output, inputs, ctx, plan, workspace);
        break;
    case DataType::s8:
        conv2d_channel_last_int8_impl<TAct, int8_t>(attrs, output, inputs, ctx, plan, workspace);
        break;
    case DataType::u8:
        conv2d_channel_last_int8_impl<TAct, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
        break;
    default:
        conv2d_channel_last_int8_impl<TAct, float>(attrs, output, inputs, ctx, plan, workspace);
        break;
    }
}

// =========================================================================
//  Plan
// =========================================================================

Conv2DGemmPlan make_plan(const Conv2DAttributes& attrs,
                        std::span<const TensorDesc> inputs,
                        std::span<const TensorDesc> outputs,
                        int num_threads,
                        bool use_thread_slots,
                        int64_t num_slots)
{
    NNOPS_ASSERT(inputs.size() >= 2);
    NNOPS_ASSERT(outputs.size() >= 1);

    const auto& in  = inputs[0];
    const auto& wt  = inputs[1];
    const auto& out = outputs[0];

    Conv2DGemmPlan plan;
    plan.route = Conv2DGemmRoute::ChannelLast;

    const DataType dt = in.dtype;
    const bool quant = is_quantized_conv2d(wt.dtype);

    if (quant) {
        // Activation: pre-quantized (u8/s8), or fp (quantized per call).
        if (dt != DataType::u8 && dt != DataType::s8 &&
            dt != DataType::f32 && dt != DataType::f16) {
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
        if (dt != DataType::f32 && dt != DataType::f16) {
            return plan;
        }
        if (wt.dtype != dt || out.dtype != dt) {
            return plan;
        }
        if (inputs.size() > 2 && inputs[2].dtype != dt) {
            return plan;
        }
    }

    const int64_t N  = in.dims[0];
    const int64_t IC = in.dims[1];
    // The channel-last weight is TensorFlow's [KH, KW, IC/G, OC]; anything else
    // is not this route's — except the prepacked blob, whose shape leads with OC
    // (see conv_packed_weight_desc).
    const bool prepacked = wt.layout == TensorLayout::PackedHWIO;
    if (!is_channel_last_weight(wt.layout) && !prepacked) { return plan; }
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

    // The block sizes are the prepacked filter's geometry: conv_packed_weight_last
    // resolves them from the attributes and the filter's dims alone, and a plan
    // choosing again would size panels the blob does not hold.
    const ConvPackedWeightLast pg =
        conv_packed_weight_last(attrs, ic_per_group, oc_per_group, wt.dtype);
    plan.icn_block = pg.icn_block;
    plan.ocn_block = pg.ocn_block;
    const int64_t nr_for_split = pg.nr;
    plan.m_block   = kMBlock;
    plan.oh_chunks = 1;
    plan.oc_chunks = 1;

    const int64_t NG = N * G;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    const int64_t esz = quant ? 1 : static_cast<int64_t>(data_type_size(dt));

    // Rows first (the axis every task shares), then the filter axis when the
    // rows cannot fill the pool — the planar plan's order and nearest-round rule.
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
    if (2 * plane_tasks < nt) {
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
               ((OH + plan.oh_chunks - 1) / plan.oh_chunks) * OW * plan.ocn_block
                   > kTaskTileCap) {
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
    plan.ph_block = std::clamp<int64_t>(
        kPassAccBytes / (pass_esz * plan.ocn_block * std::max<int64_t>(OW, 1)),
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
    plan.a_pack_stride = quant
        ? gemm_align64(static_cast<int64_t>(num_panels_max(static_cast<int>(plan.m_block), MR_I8))
                       * geom_ldd_i8_a((kdim_max + 3) & ~int64_t{3}))
        : 0;
    plan.a_size = plan.a_pack_stride
                + ((plan.pointwise && !pw_gather)
                       ? 0
                       : plan.m_block * (quant ? ((kdim_max + 3) & ~int64_t{3})
                                               : kdim_max + kABSlack));
    // In HWIO a panel's k row is one OC-length run, nothing transposed. Only a
    // multi-tap filter needs the staging tile, and only when a task does pack —
    // which neither a prepacked filter nor a shared per-call one does.
    plan.stage_size = (karea > 1 && plan.pack_size > 0) ? nr_max * kdim_max : 0;
    const int64_t pass_tiles = plan.ph_block * OW;
    plan.orig_size = keep_orig ? pass_tiles * plan.ocn_block : 0;
    // The int32 accumulator is a per-slot tile set shaped like the residual.
    plan.acc_size = quant ? pass_tiles * plan.ocn_block : 0;
    // An fp activation is materialized into a compact s8 NHWC plane once per
    // call, as on the planar route. A pre-quantized u8 one is materialized too
    // when it takes the pointwise route, whose in-row A read cannot shift bytes.
    const bool fp_act = (dt == DataType::f32 || dt == DataType::f16);
    plan.quant_size = (quant && (fp_act || (dt == DataType::u8 && plan.pointwise)))
                    ? N * in.dims[2] * in.dims[3] * IC : 0;

    const int64_t total = NG * plan.oh_chunks * plan.oc_chunks;
    const int64_t avail = (num_slots > 0) ? num_slots : nt;
    plan.num_slots = (use_thread_slots && avail < total) ? avail : total;

    plan.workspace_size = static_cast<size_t>(gemm_scratch_layout(plan, esz).total);
    return plan;
}


// ---- dispatch -------------------------------------------------------------

/// The quantized route's dtype dispatch; the plan already refused the rest.
Status conv2d_channel_last_int8_run(const Conv2DAttributes& attrs, TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            const Conv2DGemmPlan& plan, void* workspace)
{
    switch (inputs[0].data_type()) {
    case DataType::f32: conv2d_channel_last_int8_out<float>(attrs, output, inputs, ctx, plan, workspace); break;
    case DataType::f16: conv2d_channel_last_int8_out<half>(attrs, output, inputs, ctx, plan, workspace); break;
    case DataType::u8:  conv2d_channel_last_int8_out<uint8_t>(attrs, output, inputs, ctx, plan, workspace); break;
    case DataType::s8:  conv2d_channel_last_int8_out<int8_t>(attrs, output, inputs, ctx, plan, workspace); break;
    default: return Status::unsupported;
    }
    return Status::ok;
}

/// The channel-last driver: quantized weights first, then the two fp element types.
Status run(const Conv2DAttributes& attrs, TensorView& output,
           std::span<const TensorView> inputs, const ComputeContext& ctx,
           const Conv2DGemmPlan& plan, void* workspace)
{
    if (is_quantized_conv2d(inputs[1].data_type())) {
        return conv2d_channel_last_int8_run(attrs, output, inputs, ctx, plan, workspace);
    }
    switch (inputs[0].data_type()) {
    case DataType::f32:
        conv2d_channel_last_impl<float>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    case DataType::f16:
        conv2d_channel_last_impl<half>(attrs, output, inputs, ctx, plan, workspace);
        return Status::ok;
    default:
        return Status::unsupported;
    }
}

}  // anonymous namespace
}  // namespace channel_last

// =========================================================================
//  Public API — one route per axis, chosen from the activation's layout
// =========================================================================

namespace {

/// A route is (make_plan, run); both are template parameters to conv2d_gemm_run
/// so the plan/pool/dispatch prologue is written once.
struct PlanarRoute {
    static Conv2DGemmPlan make_plan(const Conv2DAttributes& attrs,
                                    std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs,
                                    int num_threads, bool use_thread_slots,
                                    int64_t num_slots)
    {
        return planar::make_plan(attrs, inputs, outputs, num_threads,
                                 use_thread_slots, num_slots);
    }
    static Status run(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx, const Conv2DGemmPlan& plan,
                      void* workspace)
    {
        return planar::run(attrs, output, inputs, ctx, plan, workspace);
    }
};

struct ChannelLastRoute {
    static Conv2DGemmPlan make_plan(const Conv2DAttributes& attrs,
                                    std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs,
                                    int num_threads, bool use_thread_slots,
                                    int64_t num_slots)
    {
        return channel_last::make_plan(attrs, inputs, outputs, num_threads,
                                       use_thread_slots, num_slots);
    }
    static Status run(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx, const Conv2DGemmPlan& plan,
                      void* workspace)
    {
        return channel_last::run(attrs, output, inputs, ctx, plan, workspace);
    }
};

/// The shared prologue: descriptors from the views, one plan, one pooled arena,
/// then the route's driver. A zeroed plan (icn_block 0) is the "unsupported
/// dtype combination" answer, refused before anything is allocated.
template <class TRoute>
Status conv2d_gemm_run(const Conv2DAttributes& attrs, TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx)
{
    std::vector<TensorDesc> descs;
    descs.reserve(inputs.size());
    for (const auto& t : inputs) { descs.push_back(t.desc()); }
    const TensorDesc outs[] = {output.desc()};

    const Conv2DGemmPlan plan = TRoute::make_plan(
        attrs, descs, outs, ctx.cpu.thread_count(), ctx.cpu.thread_slot_count() > 0,
        ctx.cpu.thread_slot_count());
    if (plan.icn_block <= 0) { return Status::unsupported; }

    // A pointwise call with no add_to needs no scratch, but the pool rejects a
    // zero-byte request: ask for the smallest arena instead.
    PoolPtr scratch(plan.workspace_size ? plan.workspace_size : 1);
    if (!scratch) { return Status::out_of_memory; }

    return TRoute::run(attrs, output, inputs, ctx, plan, scratch.get());
}

}  // anonymous namespace

// =========================================================================
//  The prepacked filter (Conv2D::prepackWeights, im2col route)
// =========================================================================
//
// The pack compute() otherwise does is a function of the filter alone, so it is
// a load-time artifact: query a descriptor, allocate, fill once — the same
// two-step contract as the Winograd U. It is safe because the geometry comes from
// the attributes plus the filter's own dims, never the activation (hence these
// take channel counts, not a plan), and the plan adopts icn_block/ocn_block from
// conv_packed_weight_last / _planar rather than choosing again — so "the panels
// prepackWeights built" and "the panels the plan sized for" are one decision.
//
// The channel-last blob is addressed by *panel*: panel p holds output channels
// [p·nr, (p+1)·nr) of one k block, for every k block of the group — so any
// panel-aligned oc range is readable and the plan's N tile stops constraining it.

ConvPackedWeightLast conv_packed_weight_last(const Conv2DAttributes& attrs,
                                             int64_t ic_per_group,
                                             int64_t oc_per_group,
                                             DataType weight_dtype)
{
    ConvPackedWeightLast g;
    if (ic_per_group <= 0 || oc_per_group <= 0 || attrs.groups <= 0) { return g; }

    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];
    const bool quant = is_quantized_conv2d(weight_dtype);
    const int64_t esz = quant ? 1 : static_cast<int64_t>(data_type_size(weight_dtype));

    // Block sizes factorised here so the plan has one place to read them from and
    // prepackWeights can resolve them without an activation. A 1x1 filter at unit
    // stride runs K over the whole channel count in one pass; padding is not
    // consulted (no input shape), and a padded 1x1 gathers with the same wide K.
    const bool unit_1x1 = karea == 1 && attrs.stride[0] == 1 && attrs.stride[1] == 1;
    g.nr = quant ? NR_MAX_I8
                 : (weight_dtype == DataType::f16 ? NR_MAX_F16 : NR_MAX_F32);
    g.icn_block = unit_1x1
        ? ic_per_group
        : std::min<int64_t>(channel_last::get_ic_block_nhwc(karea, ic_per_group),
                            ic_per_group);
    // Balance the oc blocks rather than taking kOcnBlock and leaving a sliver:
    // 144 is 80 + 64, not 128 + 16 (an eighth-full block paying every per-block
    // cost: accumulator fill, A pack, epilogue).
    const int64_t oc_blocks_wanted = oc_per_group <= channel_last::kOcnSingleCap
        ? 1 : (oc_per_group + channel_last::kOcnBlock - 1) / channel_last::kOcnBlock;
    const int64_t per_block =
        (oc_per_group + oc_blocks_wanted - 1) / oc_blocks_wanted;
    g.ocn_block = std::min<int64_t>(
        oc_per_group, ((per_block + g.nr - 1) / g.nr) * g.nr);

    g.kdim_max    = g.icn_block * karea;
    g.panels_oc   = (oc_per_group + g.nr - 1) / g.nr;
    g.slot_full   = gemm_panel_slot(g.kdim_max, g.nr, esz, quant);
    g.nfull_ic    = ic_per_group / g.icn_block;
    g.tail_ic     = ic_per_group - g.nfull_ic * g.icn_block;
    g.slot_tail   = g.tail_ic
        ? gemm_panel_slot(g.tail_ic * karea, g.nr, esz, quant) : 0;
    g.group_elems = g.panels_oc
                  * (g.nfull_ic * g.slot_full + (g.tail_ic ? g.slot_tail : 0));
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

ConvPackedWeightPlanar conv_packed_weight_planar(const Conv2DAttributes& attrs,
                                                 int64_t ic_per_group,
                                                 int64_t oc_per_group,
                                                 DataType weight_dtype)
{
    ConvPackedWeightPlanar g;
    if (ic_per_group <= 0 || oc_per_group <= 0 || attrs.groups <= 0) { return g; }
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];

    // A float filter is already the planar GEMM's A operand (OIHW is exactly
    // that), so the "prepacked" form is only a densified copy. Its one job is to
    // make the contract total: every supported filter has one prepacked form.
    if (!is_quantized_conv2d(weight_dtype)) {
        g.dense = true;
        g.total = attrs.groups * oc_per_group * ic_per_group * karea;
        return g;
    }

    // The planar plan's own blocks (planar::make_plan), so the A panels it
    // otherwise rebuilds per task are the ones the blob holds. A 1x1 filter at
    // unit stride runs K over the whole channel count in one pass, the plan's
    // pointwise rule; padding is not consulted, for the channel-last reason.
    const bool unit_1x1 = karea == 1 && attrs.stride[0] == 1 && attrs.stride[1] == 1;
    g.icn_block = unit_1x1
        ? ic_per_group
        : std::min<int64_t>(planar::get_ic_block(static_cast<int>(karea)), ic_per_group);
    g.ocn_block = std::min<int64_t>(planar::kOcnBlock, oc_per_group);
    g.kdim_max  = g.icn_block * karea;
    g.num_oc_blocks = (oc_per_group + g.ocn_block - 1) / g.ocn_block;
    g.nfull_ic  = ic_per_group / g.icn_block;
    g.tail_ic   = ic_per_group - g.nfull_ic * g.icn_block;
    const int64_t panels = num_panels_max(static_cast<int>(g.ocn_block), MR_I8);
    g.set_full = panels * geom_ldd_i8_a(g.kdim_max);
    g.set_tail = g.tail_ic ? panels * geom_ldd_i8_a(g.tail_ic * karea) : 0;
    g.group_elems = (g.nfull_ic + (g.tail_ic ? 1 : 0)) * g.num_oc_blocks * g.set_full;
    g.total = attrs.groups * g.group_elems;
    // The R_a table trails the panels (see ConvPackedWeightLast::sums_off).
    g.sums_off = g.total;
    const int64_t sum_bytes =
        attrs.groups * oc_per_group * static_cast<int64_t>(sizeof(int32_t));
    g.total += (sum_bytes + 3) & ~int64_t{3};
    return g;
}

int64_t ConvPackedWeightPlanar::offset(int64_t g, int64_t ic_block,
                                       int64_t oc_block) const noexcept
{
    const int64_t ic = ic_block * num_oc_blocks * set_full;
    const int64_t oc = oc_block * set_full;
    return g * group_elems + ic + oc;
}

namespace {

/// The blob's descriptor: [OC, ⌈elems / OC⌉] with an explicit pitch, so numel()
/// is exactly the buffer — how the two-step prepack's callers size their
/// allocation. The dims after [0] carry no meaning.
TensorDesc packed_weight_desc(int64_t oc_per_group, int64_t groups,
                              int64_t elems, DataType dtype, TensorLayout layout)
{
    TensorDesc d;
    if (elems <= 0 || oc_per_group <= 0) { return d; }
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

}  // anonymous namespace

TensorDesc conv_packed_weight_desc(const Conv2DAttributes& attrs,
                                   const TensorDesc& weight)
{
    // The filter's own dims give the channel counts: OIHW is [OC][IC/G][KH][KW],
    // HWIO is [KH][KW][IC/G][OC].
    const bool hwio = is_channel_last_weight(weight.layout);
    if (weight.rank != 4 || (!hwio && weight.layout != TensorLayout::NCHW)) {
        return TensorDesc{};
    }
    const int64_t OC  = hwio ? weight.dims[3] : weight.dims[0];
    const int64_t ICG = hwio ? weight.dims[2] : weight.dims[1];
    const int64_t G   = attrs.groups;
    if (G <= 0 || OC % G != 0 || ICG <= 0) { return TensorDesc{}; }

    if (hwio) {
        const ConvPackedWeightLast g =
            conv_packed_weight_last(attrs, ICG, OC / G, weight.dtype);
        return packed_weight_desc(OC / G, G, g.total, weight.dtype,
                                  TensorLayout::PackedHWIO);
    }
    const ConvPackedWeightPlanar g =
        conv_packed_weight_planar(attrs, ICG, OC / G, weight.dtype);
    return packed_weight_desc(OC / G, G, g.total, weight.dtype,
                              TensorLayout::PackedOIHW);
}

namespace {

/// Run @p body over [0, n) with one staging tile per worker — the channel-last
/// pack's only scratch. Falls back to one tile and a serial loop when the
/// backend reports no thread ids: a shared tile would race.
template <class Body>
void with_stage(const ComputeContext& ctx, int64_t n, int64_t stage_len, Body body)
{
    const int nslots = std::max(ctx.cpu.thread_slot_count(), 1);
    const bool par = nslots > 1 && n > 1;
    // One vector of chars keeps this helper type-agnostic.
    std::vector<char> stages(static_cast<size_t>(nslots * stage_len));
    auto run = [&](int64_t s) {
        char* const stage = stages.data() +
            (par ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : 0) * stage_len;
        body(s, stage);
    };
    if (par) {
        ctx.cpu.run(0, n, run);
    } else {
        for (int64_t s = 0; s < n; ++s) { run(s); }
    }
}

/// Pack the channel-last panels (PackedHWIO): one panel per (group, k block,
/// panel), `nr` output channels wide.
///
/// A panel is what pack_block builds for an oc run one panel wide. The task
/// split is (group, k block) — a whole panel column each, so every task writes a
/// contiguous region.
template <class T>
void pack_weight_last(const TensorView& weight, TensorView& packed,
                      const ConvPackedWeightLast& g, int64_t ic_per_group,
                      int64_t oc_per_group, int64_t groups,
                      const ComputeContext& ctx)
{
    const int64_t karea = g.kdim_max / g.icn_block;
    const int64_t num_ic_blocks = g.nfull_ic + (g.tail_ic ? 1 : 0);

    const T* const w = weight.ptr<T>();
    T* const blob = packed.ptr<T>();
    const int64_t stage_len = g.nr * g.kdim_max;   // elements

    with_stage(ctx, groups * num_ic_blocks, stage_len * static_cast<int64_t>(sizeof(T)),
               [&](int64_t s, char* stage_raw) {
        T* const stage = reinterpret_cast<T*>(stage_raw);
        const int64_t gg = s / num_ic_blocks;
        const int64_t ib = s % num_ic_blocks;

        const int64_t icn  = ib * g.icn_block;
        const int64_t icnc = std::min(g.icn_block, ic_per_group - icn);
        const int64_t kdim = icnc * karea;
        const int64_t slot = gemm_panel_slot(kdim, g.nr, sizeof(T), false);
        T* const col = blob + g.offset(gg, ib);

        // HWIO: one input channel's tap is one OC-length run; the group's output
        // channels start at gg·oc_per_group.
        const int64_t w_ic  = weight.shape(3);          // between input channels
        const int64_t w_tap = ic_per_group * w_ic;      // between filter taps
        for (int64_t p = 0; p < g.panels_oc; ++p) {
            const int64_t ocn  = p * g.nr;
            const int64_t ocnc = std::min(g.nr, oc_per_group - ocn);
            channel_last::pack_block<T>(col + p * slot, slot, stage,
                          w + gg * oc_per_group + ocn + icn * w_ic,
                          w_ic, w_tap, ocnc, icnc, static_cast<int>(karea),
                          g.nr, static_cast<int>(kdim));
        }
    });
}

/// pack_block_i8's sibling of the float pack above: same split, same loop.
void pack_weight_last_i8(const TensorView& weight, TensorView& packed,
                         const ConvPackedWeightLast& g, int64_t ic_per_group,
                         int64_t oc_per_group, int64_t groups,
                         const ComputeContext& ctx)
{
    const int64_t karea = g.kdim_max / g.icn_block;
    const int64_t num_ic_blocks = g.nfull_ic + (g.tail_ic ? 1 : 0);
    const bool a_biased = arch::mma_i8_folds_a_bias;

    const int8_t* const w = weight.ptr<int8_t>();
    int8_t* const blob = packed.ptr<int8_t>();
    const int64_t stage_len = g.nr * g.kdim_max;   // bytes

    with_stage(ctx, groups * num_ic_blocks, stage_len, [&](int64_t s, char* stage_raw) {
        int8_t* const stage = reinterpret_cast<int8_t*>(stage_raw);
        const int64_t gg = s / num_ic_blocks;
        const int64_t ib = s % num_ic_blocks;

        const int64_t icn  = ib * g.icn_block;
        const int64_t icnc = std::min(g.icn_block, ic_per_group - icn);
        const int64_t kdim = icnc * karea;
        const int64_t slot = gemm_panel_slot(kdim, g.nr, 1, true);
        int8_t* const col = blob + g.offset(gg, ib);

        const int64_t w_ic  = weight.shape(3);
        const int64_t w_tap = ic_per_group * w_ic;
        for (int64_t p = 0; p < g.panels_oc; ++p) {
            const int64_t ocn  = p * g.nr;
            const int64_t ocnc = std::min(g.nr, oc_per_group - ocn);
            channel_last::pack_block_i8(col + p * slot, slot, stage,
                          w + gg * oc_per_group + ocn + icn * w_ic,
                          w_ic, w_tap, ocnc, icnc, static_cast<int>(karea),
                          g.nr, static_cast<int>(kdim), a_biased);
        }
    });
}

/// The quantized planar A panels (PackedOIHW): one set per (group, k block, oc
/// block), each `num_panels_max(ocn_block, MR_I8)` panels of MR_I8 output
/// channels at the k block's own A stride.
void pack_weight_planar_i8(const TensorView& weight, TensorView& packed,
                           const ConvPackedWeightPlanar& g,
                           int64_t ic_per_group, int64_t oc_per_group,
                           int64_t groups)
{
    const int64_t karea = g.kdim_max / g.icn_block;
    const bool a_biased = arch::mma_i8_folds_a_bias;
    const int64_t num_ic_blocks = g.nfull_ic + (g.tail_ic ? 1 : 0);
    const int64_t oc_blocks = g.num_oc_blocks;
    const int64_t w_ocn_step = ic_per_group * karea;

    const int8_t* const w = weight.ptr<int8_t>();
    int8_t* const blob = packed.ptr<int8_t>();

    for (int64_t gg = 0; gg < groups; ++gg) {
        for (int64_t ib = 0; ib < num_ic_blocks; ++ib) {
            const int64_t icn  = ib * g.icn_block;
            const int64_t icnc = std::min(g.icn_block, ic_per_group - icn);
            const int64_t kdim = icnc * karea;
            for (int64_t ob = 0; ob < oc_blocks; ++ob) {
                const int64_t ocn  = ob * g.ocn_block;
                const int64_t ocnc = std::min(g.ocn_block, oc_per_group - ocn);
                int8_t* const dst = blob + g.offset(gg, ib, ob);
                tile_pack_lhs_i8(false, static_cast<int>(ocnc), static_cast<int>(kdim),
                                 dst, static_cast<int>(geom_ldd_i8_a(kdim)),
                                 w + gg * oc_per_group * w_ocn_step +
                                     icn * karea + ocn * w_ocn_step,
                                 static_cast<int>(w_ocn_step), a_biased);
            }
        }
    }
}

/// Σ_k weight[oc, k] per output channel over a group's whole K — the table the
/// int8 epilogues compensate with (R_b channel-last, R_a planar). Written from
/// the *raw* filter, in whichever order its layout makes contiguous.
void write_weight_sums(const int8_t* w, int64_t OC, int64_t ic_per_group,
                       int64_t karea, bool hwio, const TensorView& weight,
                       int32_t* out)
{
    if (hwio) {
        // [KH][KW][IC/G][OC]: the walk covers the group's input channels and
        // *all* groups' output channels at once — the OC axis spans groups.
        const int64_t w_ic  = weight.shape(3);
        const int64_t w_tap = ic_per_group * w_ic;
        for (int64_t t = 0; t < karea; ++t) {
            for (int64_t i = 0; i < ic_per_group; ++i) {
                const int8_t* const row = w + t * w_tap + i * w_ic;
                for (int64_t oc = 0; oc < OC; ++oc) { out[oc] += row[oc]; }
            }
        }
        return;
    }
    // [OC][IC/G·KH·KW]: one output channel's whole K is one contiguous run.
    const int64_t k_full = ic_per_group * karea;
    for (int64_t oc = 0; oc < OC; ++oc) {
        out[oc] = nnops::kernel::int8_row_sum<int8_t>(w + oc * k_full, k_full);
    }
}

}  // anonymous namespace

void conv_packed_weight_pack(const Conv2DAttributes& attrs,
                             const TensorView& weight,
                             TensorView& packed,
                             const ComputeContext& ctx)
{
    const bool hwio = is_channel_last_weight(weight.layout());
    const int64_t OC  = hwio ? weight.shape(3) : weight.shape(0);
    const int64_t ICG = hwio ? weight.shape(2) : weight.shape(1);
    const int64_t G   = attrs.groups;
    const int64_t OCG = OC / G;
    const DataType dtype = weight.data_type();
    const int64_t esz = static_cast<int64_t>(data_type_size(dtype));
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];

    // No zero-fill here: the panel packers mask-load a short panel and zero-fill
    // the lanes past its width, so every byte a consumer reads is written.
    if (hwio) {
        const ConvPackedWeightLast g = conv_packed_weight_last(attrs, ICG, OCG, dtype);
        NNOPS_ASSERT(packed.numel() >= g.total &&
                     "conv_packed_weight_pack: buffer smaller than the panels");
        switch (dtype) {
        case DataType::f32:
            pack_weight_last<float>(weight, packed, g, ICG, OCG, G, ctx);
            break;
        case DataType::f16:
            pack_weight_last<half>(weight, packed, g, ICG, OCG, G, ctx);
            break;
        case DataType::s8:
            pack_weight_last_i8(weight, packed, g, ICG, OCG, G, ctx);
            break;
        default:
            NNOPS_ASSERT(!"conv_packed_weight_pack: unsupported filter dtype");
            return;
        }
        if (is_quantized_conv2d(dtype)) {
            // The sums land on an unaligned byte offset: accumulate whole, copy in.
            std::vector<int32_t> sums(static_cast<size_t>(OC), 0);
            write_weight_sums(weight.ptr<int8_t>(), OC, ICG, karea, true, weight,
                              sums.data());
            std::memcpy(packed.ptr<char>() + g.sums_off, sums.data(),
                        static_cast<size_t>(OC) * sizeof(int32_t));
        }
        return;
    }

    const ConvPackedWeightPlanar g = conv_packed_weight_planar(attrs, ICG, OCG, dtype);
    NNOPS_ASSERT(packed.numel() >= g.total &&
                 "conv_packed_weight_pack: buffer smaller than the panels");
    if (g.dense) {
        // A planar float filter is already the GEMM's A operand: the blob is that
        // matrix, and the copy is the whole pack (and how any pitch ends up dense).
        std::memcpy(packed.ptr<char>(), weight.ptr<char>(),
                    static_cast<size_t>(g.total) * static_cast<size_t>(esz));
    } else {
        pack_weight_planar_i8(weight, packed, g, ICG, OCG, G);
        std::vector<int32_t> sums(static_cast<size_t>(OC), 0);
        write_weight_sums(weight.ptr<int8_t>(), OC, ICG, karea, false, weight,
                          sums.data());
        std::memcpy(packed.ptr<char>() + g.sums_off, sums.data(),
                    static_cast<size_t>(OC) * sizeof(int32_t));
    }
}

Conv2DGemmPlan get_conv2d_implicit_gemm_plan(const Conv2DAttributes& attrs,
                                             std::span<const TensorDesc> inputs,
                                             std::span<const TensorDesc> outputs,
                                             int num_threads,
                                             bool use_thread_slots,
                                             int64_t num_slots)
{
    if (is_channel_last(inputs[0].layout)) {
        return ChannelLastRoute::make_plan(attrs, inputs, outputs, num_threads,
                                           use_thread_slots, num_slots);
    }
    return PlanarRoute::make_plan(attrs, inputs, outputs, num_threads,
                                  use_thread_slots, num_slots);
}

Status conv2d_implicit_gemm_kernel(const Conv2DAttributes& attrs,
                                   TensorView& output,
                                   std::span<const TensorView> inputs,
                                   const ComputeContext& ctx,
                                   void* /*workspace*/)
{
    if (is_channel_last(inputs[0].layout())) {
        return conv2d_gemm_run<ChannelLastRoute>(attrs, output, inputs, ctx);
    }
    return conv2d_gemm_run<PlanarRoute>(attrs, output, inputs, ctx);
}

}  // namespace nnops::backend::cpu
