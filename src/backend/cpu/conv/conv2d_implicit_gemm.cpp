/// @file conv2d_implicit_gemm.cpp
/// @brief Conv2D as implicit GEMM — both memory layouts, one driver.
///
/// conv2d_implicit_gemm.h describes the two mappings and everything they share
/// (one plan, one scratch geometry, one entry). This file is the two routes, one
/// per half:
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
/// The route comes from the activation's layout in the entry
/// (conv2d_implicit_gemm_kernel), which owns the plan/pool/dispatch prologue the
/// two share and hands each route its plan. Everything inside a route is
/// file-local — each half opens its own namespace, so their helpers (tile_save,
/// pack_block, …) never have to carry a route prefix.
///
/// The parallel decomposition is the same on both: (batch, group) × oh-chunks ×
/// oc-chunks, sized by the plan. Output tiles are disjoint, so every route is
/// bit-identical to serial execution.

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

/// Input-channel block size from the kernel area, sized so one GEMM call sees
/// kdim = ic_block * karea of roughly 144-200 whatever the kernel size.
///
/// The kdim this produces is the block's real cost knob, not its budget: every
/// ic-block re-reads and re-writes the whole output tile (the accumulator *is*
/// the output), so a short K pays that pass too often, and the pack/gather
/// restart per block. On a 3x3 the 8 rung gave kdim 72 and a 16 rung (kdim 144)
/// measured -2.5..-4.4% interleaved at one thread; the 5x5 and larger rungs move
/// much less (-0.5%) because their kdim was already in range. Above that the
/// col buffer it multiplies starts to cost more than the passes it saves.
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

/// Initialize an output tile (M rows of N columns, row stride ld) with the
/// per-row bias. When @p add_to, add the pre-existing output into the
/// accumulator (used for the identity-epilogue residual path).
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

    // Block sizes + parallel decomposition from the plan (single source of
    // truth — conv2d_implicit_gemm_kernel resolved them once, with the sizing).
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
    const int64_t w_ocn_step = ic_per_group * karea;

    T* out_ptr = output.ptr<T>();
    const T* in_ptr  = input.ptr<T>();
    const T* w_ptr   = weight.ptr<T>();
    const T* b_ptr   = has_bias ? inputs[2].ptr<T>() : nullptr;

    // Scratch slices: one per worker thread (thread_id-indexed, reused across
    // every task a thread claims) when the backend reports thread ids, else one
    // per parallel task. Output tiles are disjoint either way, so the result is
    // bit-identical.
    //
    // The fp route has no col buffer: the gather writes the packed B panels
    // directly (tiled_im2col_2d_packed), so the plan sizes col_size 0 and one
    // slot is [orig][pack_b].
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

    // A pure clamp needs no epilogue pass. The MMA store already applies its
    // min/max bounds — MatMul folds Relu into them the same way (matmul.cpp's
    // attrs_n switch) — so Relu/Relu6, which is every activation in
    // MobileNetV2, costs one read+write less per output tile. Measured 11% of
    // the conv2d row at one thread, interleaved; the checkpoints are
    // bit-identical, so this is the same arithmetic in a different place.
    //
    // add_to is the exception: the residual is added *after* the activation
    // (epilogue(bias + Σ) then += orig), so it needs the tile back in memory.
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

    // The first icn block *builds* the tile instead of accumulating onto a
    // pre-filled one: the micro-kernel starts its accumulators at zero, or at
    // the bias row when there is one (tile_mma_direct's bias argument — the
    // kernel's row index is the output channel, so a bias row lands exactly
    // where tile_set_bias would have written it). MatMul takes the same
    // shortcut on its first k-block (`k == 0 && attrs.beta == 0`).
    //
    // add_to is the exception: it folds the *original* output into the tile,
    // which no accumulator seed can express, so that case still pre-fills.
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

        // Contiguous range of oh-blocks owned by this chunk (disjoint across
        // chunks, so output tiles never overlap → bit-identical to serial).
        const int64_t b0 = chunk * num_oh_blocks / oh_chunks;
        const int64_t b1 = (chunk + 1) * num_oh_blocks / oh_chunks;

        // This task's oc range, likewise disjoint. An even split of the filter
        // rows (not of the oc-block list): the task count is what the pool
        // wants, and the ranges land within one row of each other.
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
                // uniform nr_max-wide panels; every oc-block below reuses it
                // (the unified MMA route always consumes packed B).
                const int kdim = static_cast<int>(icnc * karea);
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * kdim * esz) / esz);

                if (!plan.pointwise) {
                    // Gather straight into the panels. Materializing the col
                    // matrix first and packing it afterwards moves the same
                    // bytes twice (write + re-read of roi_area*kdim elements)
                    // for a pure re-ordering; this does it in one pass. The
                    // values, zero padding and panel layout are identical.
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
                    // A pointwise gather would be a dense row copy, so building
                    // it only adds a write and a re-read; B is read from the
                    // input instead. pad == 0 and stride == 1 mean the ROI never
                    // leaves the input, so there is no border case to handle.
                    // The block's ROI inside the input plane is one roi_area-wide
                    // run per input channel, so the k stride is the channel
                    // stride rather than the ROI width. input_local already
                    // carries the batch and group offsets (above).
                    const T* b_src = input_local + oh * in_row_stride;
                    tile_pack_rhs(false, static_cast<int>(roi_area), kdim,
                                  pack_b, ldd_b, b_src, static_cast<int>(icn_step), 1.0f);
                }

                for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block) {
                    const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);
                    T* tile = tile_base + ocn * ocn_step;

                    if (icn == 0 && seed_needed) {
                        // add_to + non-identity epilogue needs the original
                        // output preserved (epilogue(bias + Σ) then += orig).
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
                    // Only the final ic-block clamps: partial sums accumulate
                    // unclamped (kblock_clamp in matmul_internal.h).
                    const bool clamp_here = is_last_icn && epilogue_folded;
                    // An unseeded first block overwrites C outright; every
                    // later block (and every seeded one) accumulates.
                    tile_mma_direct(static_cast<int>(ocnc), static_cast<int>(roi_area),
                                            kdim,
                                            tile, static_cast<int>(ocn_step),
                                            a, static_cast<int>(w_ocn_step),
                                            pack_b,
                                            clamp_here ? clamp_lo : -kInf,
                                            clamp_here ? clamp_hi : kInf,
                                            /*zero_mode=*/icn == 0 && !seed_needed,
                                            // The bias belongs to the block that
                                            // builds the tile; a later one would
                                            // re-seed and drop the partial sum.
                                            (icn == 0 && bias_in_mma) ? bias_ptr + ocn : nullptr);

                    // An identity epilogue is a whole-tile copy-back — skip it
                    // (matmul.cpp guards the same call the same way).
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
// The same product with integer operands: A is the s8 weight (per-channel
// QuantParams), B the u8/s8 im2col activation, the accumulator int32, and the
// epilogue compensates the zero points and either dequantizes (fp destination)
// or requantizes (s8/u8 destination). An fp activation is quantized into a
// compact plane once per call, before the tile loop, and is bytes from then on
// — quantizing inside the gather would repeat it once per oc block, and only a
// materialized plane lets a 1x1 convolution take the pointwise route.

/// True when every zero point the tensor carries equals @p v — i.e. the
/// compensation this tensor drives is the identity on every row.
inline bool all_zp_equal(const QuantParams& q, int64_t n, int32_t v) noexcept {
    if (q.zero_point_data == nullptr) { return q.zero_point == v; }
    for (int64_t i = 0; i < n; ++i) {
        if (q.zero_point_data[i] != v) { return false; }
    }
    return true;
}

/// Quantize an fp activation into the compact int8 plane the rest of the kernel
/// reads. Dense [N][IC][IH][IW]; the source is walked through its row pitch.
///
/// This is one linear pass at ~17 Gvalues/s (the same arithmetic
/// `quant_kernel::quantization` performs everywhere else), against the
/// alternative of quantizing inside every im2col gather — which repeats the work
/// once per oc-block and once per oh-block that re-reads the same input. The
/// plane also lets an fp 1x1 conv take the pointwise route, which needs a dense
/// quantized plane to read.
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
        // Pitched source: quantize row by row, never across the pitch tail.
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

    // Below this a dispatch costs more than the pass: the pool re-spawns its
    // workers per run, which is ~100 us at 4 threads on this host against ~10 us
    // of work for a 200k-element plane. Small planes stay on the calling
    // thread; big ones (a wide 56x56 layer is 400k elements) still split.
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

/// The quantized implementation for one (activation, storage, output) dtype
/// triple. @p TDst is the integer storage the GEMM consumes: the activation's
/// own type on the pre-quantized route, `act_quant_dtype` on the fp one.
template <class TAct, class TDst, class TOut>
void conv2d_planar_int8_impl(const Conv2DAttributes& attrs,
                             TensorView& output,
                             std::span<const TensorView> inputs,
                             const ComputeContext& ctx,
                             const Conv2DGemmPlan& plan,
                             void* workspace)
{
    // f32/f16 activations are quantized into the plane before the tile loop
    // (quantize_plane); u8/s8 already arrive in the storage type.
    constexpr bool kFpAct = !std::is_same_v<TAct, int8_t> && !std::is_same_v<TAct, uint8_t>;
    /// The B packer XORs 0x80, i.e. reads the stored bytes as unsigned and
    /// shifts them to (value − 128).
    constexpr bool kU8 = std::is_same_v<TDst, uint8_t>;
    /// An integer destination: the accumulators are requantized (not
    /// dequantized) and the bias is an int32 per output channel.
    constexpr bool kIntOut = std::is_same_v<TOut, int8_t> || std::is_same_v<TOut, uint8_t>;

    const auto& input  = inputs[0];
    const auto& weight = inputs[1];
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);

    // Weight: [OC, IC/G, KH, KW], compact.
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

    // Block sizes + parallel decomposition from the plan (single source of
    // truth — conv2d_int8_kernel resolved them once, with the sizing).
    const int64_t icn_block = plan.icn_block;
    const int64_t ocn_block = plan.ocn_block;
    const int64_t oh_block  = plan.oh_block;
    const int64_t num_oh_blocks = plan.num_oh_blocks;
    const int64_t oh_chunks     = plan.oh_chunks;

    const int64_t ocn_step = OH * out_row_stride;        // between output channels
    const int64_t ob_step  = OC * ocn_step;              // between batch samples
    const int64_t og_step  = oc_per_group * ocn_step;    // between groups (output)

    // Weight row stride within a group: [oc_per_group][ic_per_group][karea].
    const int64_t w_ocn_step = ic_per_group * karea;

    // ---- quantization parameters -----------------------------------------
    const QuantParams& qa = input.quant_params();
    const QuantParams& qw = weight.quant_params();

    // Per-tensor activation parameters (offline: fixed before the call, and in
    // force whether the activation arrives quantized or as fp).
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
    // weight's zero point is not already the compensated one. Everything else
    // is per output channel, so it is computed once per call — into the bias for
    // an integer destination (what MLAS prepacks into its Bias), into an int32
    // row offset for the fp one (see row_fold / bias_fold below).
    const bool need_col_sum = !all_zp_equal(qw, OC, -a_bias_offset);
    // r_a feeds both the column term and the fold; the fp epilogue reads both
    // reductions whenever it compensates.
    const bool need_comp = need_col_sum || zp_eff != 0;

    // ---- the activation as stored bytes -----------------------------------
    // An fp activation is quantized into the plane at the front of the scratch
    // once, here, and from that point on the call is indistinguishable from the
    // pre-quantized route: the gather copies bytes, the pad value is the zero
    // point, and an fp 1x1 conv reaches pointwise like any other.
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
    // The bias is typed by the branch: output dtype on the fp destination,
    // int32 [OC] on the integer one.
    [[maybe_unused]] const TOut* b_ptr =
        (has_bias && !kIntOut) ? inputs[2].ptr<TOut>() : nullptr;
    [[maybe_unused]] const int32_t* b_i32 =
        (has_bias && kIntOut) ? inputs[2].ptr<int32_t>() : nullptr;
    TOut* out_ptr       = output.ptr<TOut>();
    // The requant's output parameters, read only for an integer destination.
    [[maybe_unused]] const QuantParams& qc = output.quant_params();

    // R_a[oc] = Σ_k weight[oc,k], over the whole K (never a single ic-block).
    // The weight layout is [OC][IC/G][KH][KW], so the global output-channel
    // index works across groups.
    //
    // Deliberately serial: it is one SIMD pass over the weight panel — the same
    // bytes the A pack streams in every ic-block, so a few percent of the call —
    // while a `ctx.cpu.run` here is a second pool dispatch on top of the tile
    // one. With the pool that spawns and joins per parallel_for that dispatch
    // costs more than the pass itself and grows with the pool: measured on this
    // host as 1.18x at 2T -> 1.9x at 20T on a 3x3 whose zero points are not the
    // packer's (bench_conv2d conv2d_i8_act_zp_ab, the zp120 arms).
    std::vector<int32_t> r_a;
    if (need_comp) {
        r_a.resize(static_cast<size_t>(OC));
        for (int64_t oc = 0; oc < OC; ++oc) {
            r_a[static_cast<size_t>(oc)] =
                k::int8_row_sum<int8_t>(w_ptr + oc * w_ocn_step, k_full);
        }
    }

    // The per-channel half of the compensation, computed once per call instead
    // of once per element: −zp_b·(r_a − k·zp_a) per output channel. An integer
    // destination folds it into the bias; an fp one takes it as an int32 row
    // offset ahead of the dequantize (its own bias is fp and cannot carry it).
    // Either way the epilogue then compensates only for the column term, and
    // sweeps r_b only when that term exists.
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

    // R_b[column] = Σ_k (stored byte) over the whole K of one output column,
    // accumulated across ic-blocks. Plain vector: it is per slot and per
    // oh-block, and never outlives this call.
    const int64_t r_b_len = plan.oh_block * OW;
    std::vector<int32_t> r_b_buf(
        need_col_sum ? static_cast<size_t>(plan.num_slots * r_b_len) : 0);

    // ---- scratch slices ---------------------------------------------------
    // The quantized-activation plane, when there is one, occupies the front of
    // the arena (plan.quant_size bytes, shared across slots); col_data, orig and
    // pack_b follow it per slot, and acc / pack_a sit behind the slots.
    // gemm_scratch_layout is the one place that decides all of it.
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

        // Contiguous range of oh-blocks owned by this chunk (disjoint across
        // chunks, so output tiles never overlap → bit-identical to serial).
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

            // The block is finished one oc-block at a time: the int32 accumulator
            // holds exactly one oc-block's tile, and the partial sums of an
            // oc-block must survive every ic-block that adds into them. With the
            // ic-block as the outer loop the next oc-block's fill would wipe
            // them out, so oc is the outer loop here — for the usual
            // oc_per_group <= ocn_block it is the only iteration and nothing is
            // repeated.
            //
            // When oc_per_group > ocn_block the im2col gather and B pack run once
            // per oc-block. That is a byte copy (the activation is already
            // quantized by this point, fp or not), so it stays a small part of
            // the GEMM it feeds; quantizing here instead is what used to make
            // this order expensive (bench_conv2d conv2d_i8_fpact_vs_fp_sweep).
            for (int64_t ocn = 0; ocn < oc_per_group; ocn += ocn_block) {
                const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn);

                // The column sums accumulate across this oc-block's ic-blocks.
                if (need_col_sum) {
                    std::fill(r_b, r_b + roi_area, 0);
                }

                for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                    const int64_t icnc = std::min(icn_block, ic_per_group - icn);

                    const TDst* input_local = input_ptr + icn * icn_step;
                    // pointwise: the gather would be a dense row copy of the
                    // (quantized) plane, so B is read from the input instead.
                    if (!plan.pointwise) {
                        // A padded sample is the real value 0, whose stored form
                        // is the activation's zero point.
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
                    // the pack (before the x86 panel sums are taken).
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
                        // unsigned sum into the shifted space the kernel multiplies.
                        if constexpr (kU8) {
                            k::int8_col_sum_slab<uint8_t>(b_src, b_ld, 0, roi_area, kdim, r_b);
                            for (int64_t i = 0; i < roi_area; ++i) {
                                r_b[i] -= 128 * kdim;
                            }
                        } else {
                            k::int8_col_sum_slab<int8_t>(b_src, b_ld, 0, roi_area, kdim, r_b);
                        }
                    }

                    tile_pack_lhs_i8(false, static_cast<int>(ocnc), kdim,
                                     pack_a, static_cast<int>(geom_ldd_i8_a(kdim)),
                                     weight_ptr + icn * karea + ocn * w_ocn_step,
                                     static_cast<int>(w_ocn_step), a_biased);

                    // tile_mma_pack_i8 accumulates (no zero_mode), so the first
                    // ic-block zeroes the tile instead of seeding it.
                    if (icn == 0) {
                        std::fill(acc, acc + ocnc * roi_area, 0);
                    }

                    tile_mma_pack_i8(static_cast<int>(ocnc), static_cast<int>(roi_area),
                                     kdim, acc, static_cast<int>(roi_area),
                                     pack_a, pack_b, INT32_MIN, INT32_MAX);
                }

                // The oc-block is complete: rewrite in place, on the tile, while
                // it is still hot — dequantized to fp, or requantized to the
                // integer destination.
                if constexpr (kIntOut) {
                    // MatMul's integer output pass: compensate, bias, Relu,
                    // then the output scale / zero point into s8/u8. The rows
                    // of the tile are output channels, so the weight's
                    // per-channel arrays are the per-row ones and the bias is a
                    // per-row int32.
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
                    // With the weights' zero point off the compensated one, both
                    // terms are live; otherwise the per-output-channel half is in
                    // bias_fold and only the column term would be left.
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
                    // Both terms when the weights' zero point is not the
                    // compensated one; otherwise the row half is in row_off.
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
        // A pre-quantized activation is already the storage type.
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
                        bool use_thread_slots)
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
        // Activation: pre-quantized (u8/s8), or fp quantized per im2col tile.
        if (dt != DataType::u8 && dt != DataType::s8 &&
            dt != DataType::f32 && dt != DataType::f16) {
            return plan;
        }
        // Output: an fp destination (the activation dtype when the activation
        // is fp, attributes.output_dtype when it is already quantized) or an
        // integer one (s8/u8, requantized with the output's QuantParams).
        const bool int_out = out.dtype == DataType::s8 || out.dtype == DataType::u8;
        if (!int_out && out.dtype != DataType::f32 && out.dtype != DataType::f16) {
            return plan;
        }
        if (!int_out && (dt == DataType::f32 || dt == DataType::f16) &&
            out.dtype != dt) {
            return plan;
        }
        // The bias is typed by the branch: the output dtype on an fp
        // destination, int32 on an integer one.
        const DataType bias_dt = int_out ? DataType::s32 : out.dtype;
        if (inputs.size() > 2 && inputs[2].dtype != bias_dt) {
            return plan;
        }
        // add_to needs the pre-existing output inside the accumulator; only the
        // reference implements that fold.
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

    // A 1x1 kernel at unit stride with no padding makes the im2col gather a dense
    // row copy, so the GEMM can read the input plane directly: B becomes the
    // input with ldb = the channel stride and K the whole ic_per_group. That only
    // holds while an output row's worth of input is contiguous, hence the row
    // stride test — a pitched input keeps the im2col path, which walks the pitch.
    // row_stride_elems == 0 is TensorDesc's "unknown → treat as compact"
    // convention (tensor_view.hpp:44), matching what row_stride_elems() returns
    // for the dense view at run time.
    // A fp activation is quantized into a dense int8 plane once, up front (see
    // quant_size below), so the pitch of the fp input does not disqualify
    // pointwise on the quantized route — the materialized plane is compact.
    const bool fp_act = (dt == DataType::f32 || dt == DataType::f16);
    plan.pointwise = (karea == 1 &&
                      attrs.stride[0] == 1 && attrs.stride[1] == 1 &&
                      attrs.padding[0] == 0 && attrs.padding[1] == 0 &&
                      ((quant && fp_act) ||
                       in.row_stride_elems == 0 || in.row_stride_elems == IW));

    // An fp activation is quantized into a compact plane once per call and then
    // consumed exactly like a pre-quantized one. Quantizing it in the gather
    // instead redoes the work for every oc-block (the oc-outer loop order) and
    // keeps a 1x1 conv off the pointwise route (bench_conv2d
    // conv2d_i8_fpact_vs_fp_sweep for the cost of both).
    plan.quant_size = (quant && fp_act) ? N * IC * in.dims[2] * IW : 0;

    // With no col buffer there is nothing left bounding the ic-block, so K runs
    // the whole ic_per_group in one pass — which also removes the reason
    // get_ic_block caps it at 64 for karea == 1.
    plan.icn_block = plan.pointwise
                   ? ic_per_group
                   : std::min<int64_t>(get_ic_block(static_cast<int>(karea)), ic_per_group);
    plan.ocn_block = std::min<int64_t>(kOcnBlock, oc_per_group);
    plan.oh_block  = ((PH + 1 + kAlign - 1) / kAlign) * kAlign;

    const int64_t NG = N * G;
    const int64_t nt = std::max<int64_t>(num_threads, 1);

    // The oh-block is the atomic unit of work — a chunk owns a whole number of
    // them — so the pool can never be filled past NG·num_oh_blocks tasks, no
    // matter how many threads it has. When that leaves more than half the pool
    // idle, halve the block to double the task count. The halving stops at a
    // block two n-panels wide: below that a block's GEMM is too narrow to pay
    // for re-reading its own weight panel at every oh-step.
    //
    // Measured on this host (i5-13600KF, 6P+8E, 20 threads), interleaved A/B:
    //   1x256x28x28 3x3   7 blocks -> 14: 406 -> 466 GFLOPS  (+15%, arms disjoint)
    //   1x64x56x56 3x3   14 blocks -> 20: within noise (+3%), so the gate is
    //                     "more than half the pool idle", not merely "any idle
    //                     worker" — at 8 threads the same 7-block plan against
    //                     an 8-thread pool is *hurt* by the split (397 -> 349),
    //                     and the gate is what keeps that case on 4 rows.
    // It costs no scratch in that direction: col_size shrinks with the block.
    // The expansion loop below grows it back, bounded by the whole plane.
    const int64_t nr_max = quant ? NR_MAX_I8
                                 : ((dt == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32);
    for (;;) {
        const int64_t blocks = (OH + plan.oh_block - 1) / plan.oh_block;
        if (2 * NG * blocks >= nt || plan.oh_block <= 1) { break; }
        const int64_t next = plan.oh_block / 2;
        if (next * OW < 2 * nr_max) { break; }
        plan.oh_block = next;
    }

    // Mirror of the halving loop. Once one oh-block per (batch, group) already
    // keeps at least half the pool busy the split buys no parallelism, and when
    // the weight panel is bigger than the output plane it feeds, covering the
    // whole plane means sweeping that panel once instead of once per block.
    // (MLAS's MlasConvPrepare takes its whole-plane ExpandThenGemm path on the
    // same filter-count-vs-output-size crossover, convolve.cpp:1691. Its weaker
    // proxy — filters per group vs plane area, not the full panel vs the plane —
    // is what keeps this to the small-plane / wide-filter shapes: the stronger
    // form would also catch the 1x64x56x56 1x1 blocks, which the halving
    // measurements below already put on 4 rows.) Growing past OH is not useful,
    // so the block lands on OH itself, which need not be a multiple of kAlign —
    // the kernel derives the row count per block from min(oh_block, OH - oh).
    if (oc_per_group > OH * OW) {
        while (plan.oh_block < OH) {
            const int64_t grown = std::min<int64_t>(plan.oh_block * 2, OH);
            const int64_t grown_blocks = (OH + grown - 1) / grown;
            if (2 * NG * grown_blocks < nt) { break; }  // pool still needs the split
            plan.oh_block = grown;
        }
    }

    // Grow the oh-block back while every worker still gets a whole block of its
    // own. The initial block is align(PH+1, 4) — a row count tied to the filter,
    // not to the plane — so a 28-row plane runs 4-row blocks that each restart
    // the gather, the packer and their own C read-modify-write pass. Covering
    // more rows per block measures +10..15% at one thread on the 28-plane rows
    // and within +-2% at 112 (a wash: the scratch it multiplies has caught up
    // by then), so the 28-56 row planes are what this is for.
    //
    // Two constraints keep it from losing at more threads. The block count must
    // stay a multiple of the chunk count (an uneven split leaves one worker with
    // two blocks while the rest idle: -12% at 4 threads on a 112 plane), and the
    // busiest chunk must not end up with more *rows* than the ungrown plan gave
    // it: a block size that does not divide the plane leaves a short tail, so an
    // even block count can still be an uneven row count (112 = 11x10 + 2: three
    // chunks of 30 rows and one of 22, -4.8% at 4 threads on the 112-plane 5x5;
    // 7 rows divides it into 16 balanced blocks and is accepted instead).
    // Searching the *block count* rather than doubling the row count is what
    // reaches 14 rows on a 112 plane: 4 -> 8 -> 16 skips every count that
    // divides evenly.
    if (!plan.pointwise) {
        constexpr int64_t kScratchCap = int64_t{256} << 10;   // elements per slot
        const int64_t scratch_per_row = plan.icn_block * karea * OW;
        const int64_t cap_rows = (scratch_per_row > 0) ? kScratchCap / scratch_per_row : OH;
        const int64_t max_rows = std::max<int64_t>(plan.oh_block,
                                                   std::min<int64_t>(OH, cap_rows));
        const int64_t chunks = std::clamp<int64_t>((nt + NG - 1) / NG, 1, OH);
        // Rows the busiest oh-chunk owns under block size `block` (the kernel's
        // own b0..b1 split, conv2d_implicit_gemm.cpp above).
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

    // Split the oh dimension when (batch, group) alone under-subscribes the
    // thread pool (the common batch=1 / groups=1 case would otherwise run
    // single-threaded). Each oh-chunk owns its own im2col scratch, so cap the
    // chunk count at the number of oh-blocks.
    plan.oh_chunks = std::clamp<int64_t>((nt + NG - 1) / NG, 1, plan.num_oh_blocks);

    // Second parallel axis: the filter rows. The oh split above cannot fill a
    // wide pool on a small output plane — a 7x7 plane is two oh-blocks of four
    // rows and that is the whole supply, so 20 threads run two tasks and
    // eighteen sit in the barrier (layer4 of a 224 ResNet-18 lost 155 -> 68
    // GFLOPS going from 4 threads to 20). MLAS answers this on the filter axis:
    // when the filter count outnumbers the output plane it expands the plane
    // once and hands the filter rows to the threaded GEMM (convolve.cpp:1691,
    // ExpandThenGemm), which slices M down to its own panel height; its
    // plane-splitting route (ExpandThenGemmSegmented) is for the wide case.
    // Here the plane chunks stay as they are and the filter rows become
    // additional tasks, so the grid is (batch, group) × oh-chunks × oc-chunks.
    //
    // This is the axis to spend first: each oc-chunk reads a *distinct* slice of
    // the filter, so the weight traffic is unchanged, whereas another oh-chunk
    // re-reads the same rows. What it does duplicate is the B gather — one copy
    // per oc-chunk where the single-axis loop packed one and re-read it per
    // oc-block. That is K·N against the GEMM's 2·M·N·K flops, i.e. ~1/(2·M) of
    // the work per duplicate: a few percent on the 50-row tasks below.
    //
    // Gated like the oh-block halving: split only while more than half the pool
    // is idle. A 56-plane already fills 14 threads on its own, and a split there
    // would duplicate the gather for nothing. The quantized route keeps the
    // single axis — its driver and accumulator layout is per-oh-chunk.
    //
    // The chunk count is the NEAREST whole number to nt/plane_tasks, not the
    // ceiling (the same rule, and the same reason, as the winograd plan's). An
    // odd plane count is where the difference shows: a 9-block plane at 20
    // threads wants 2 chunks (18 tasks, one wave), and ceil picks 3 (27 tasks,
    // a second wave of 7 that the barrier waits on) — measured 152 vs 226
    // GFLOPS on a 17x17 plane, so the ceiling is not a rounding detail.
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

    // The fp route fuses the gather with the pack (tiled_im2col_2d_packed), so
    // it never materializes the col matrix; the quantized route still builds
    // one and packs it separately.
    plan.col_size  = (plan.pointwise || !quant)
                   ? 0
                   : plan.icn_block * karea * plan.oh_block * OW;
    // The quantized route never reaches here with add_to (gated above), so its
    // residual buffer does not exist.
    plan.orig_size = (attrs.add_to && !quant)
                   ? plan.ocn_block * plan.oh_block * OW : 0;

    // Uniform packed-B scratch, sized at the widest n-block (a whole oh-block
    // at once) and the longest k-block. The packer's stride is computed from
    // the actual k each call, so this is a bound, not a worst case.
    //
    // The quantized route's col/pack-B scratch is in bytes whatever the
    // activation dtype is (esz 1), and its packed-B stride is the int8 panel
    // stride rather than a float panel's.
    const int64_t esz      = quant ? 1 : static_cast<int64_t>(data_type_size(dt));
    const int64_t kdim_max = plan.icn_block * karea;
    plan.pack_size = ((plan.oh_block * OW + nr_max - 1) / nr_max)
                   * gemm_panel_slot(kdim_max, nr_max, esz, quant);

    // Scratch slots: one per worker thread when the backend reports thread ids and
    // that is fewer than the task count (thread_id-indexed, reused across every
    // task a thread claims); else one per parallel task.
    const int64_t total = NG * plan.oh_chunks * plan.oc_chunks;
    plan.num_slots = (use_thread_slots && nt < total) ? nt : total;

    if (quant) {
        // The int32 accumulator holds one oc-block's whole (oc, oh-block) tile;
        // the packed-A region one oc-block's panels at the widest k-block. Both
        // sit behind the slots in the shared arena (gemm_scratch_layout).
        plan.acc_size = plan.ocn_block * plan.oh_block * OW;
        plan.pack_a_size = static_cast<int64_t>(
            num_panels_max(static_cast<int>(plan.ocn_block), MR_I8))
            * geom_ldd_i8_a(kdim_max);
    }
    plan.workspace_size = static_cast<size_t>(gemm_scratch_layout(plan, esz).total);
    return plan;
}


// ---- dispatch -------------------------------------------------------------

/// The quantized route's last dispatch level: the plan already validated the
/// dtype combination, so only the activation's own type is left to resolve.
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

/// An s8 weight selects the quantized driver whatever the activation dtype is;
/// the floating-point one has exactly the two element types the MMA has kernels
/// for. Anything else is a combination the plan already refused.
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

/// Input-channel block, sized so one GEMM call sees kdim = ic_block × karea in
/// the 144-256 range whatever the filter.
///
/// kdim is the block's real cost knob: every ic block costs one read-modify-write
/// pass over the task's output slice (the accumulator *is* the output) and
/// restarts the gather, while it also fixes the packed-B panel size
/// (nr_max × kdim) that has to stay near L1/L2. A 1x1 filter has neither a
/// gather nor a karea multiplier, so K runs four times as wide.
inline int64_t get_ic_block_nhwc(int64_t karea, int64_t ic_per_group) noexcept {
    if (karea == 1)  { return 256; }
    // A narrow channel count takes itself whole even when the ladder would split
    // it: with every channel of a pixel in the block, the taps of one filter row
    // are adjacent in memory, so the gather copies them as one run instead of
    // `KW` runs of two. That is the 3-channel stem's whole loss (IC=3, 7x7).
    if (ic_per_group * karea <= 256) { return std::max<int64_t>(ic_per_group, 1); }
    if (karea <= 9)  { return 16; }
    if (karea <= 25) { return 8;  }
    if (karea <= 49) { return 2;  }
    return 1;
}

constexpr int64_t kOcnBlock = 128;  ///< output-channel tile (the GEMM's N)
/// An oc_per_group at or below this stays one block: the N tile, its B panel and
/// the accumulator still fit the tile budget whole, and every extra oc block
/// costs a second pass over the pixels (the A pack and, on the integer route,
/// the accumulator fill and the epilogue). Splitting 144 into 80 + 64 cost the
/// int8 expand layer a quarter of its time; 256 (a real 3x3 width) still splits.
constexpr int64_t kOcnSingleCap = 192;
constexpr int64_t kMBlock   = 24;   ///< pixel tile (the GEMM's M): 4 × mr_max(6)
/// Pixels a task needs before it is worth splitting the plane again: two M
/// blocks, i.e. one full tile plus a second to hide its prologue.
constexpr int64_t kTaskMinPixels = 2 * kMBlock;
/// Bytes one accumulator pass may cover (the plan's rows-per-pass knob).
constexpr int64_t kPassAccBytes = 128 * 1024;
constexpr int64_t kABSlack  = 8;    ///< elements past the A tile (vector over-read)

/// Cap on the shared packed-weight buffer: written once per call, read by every
/// task, so a weight that fits needs no per-task pack at all — which is what
/// keeps a small plane (few pixels per task) from paying the whole weight pack
/// per task. Past 8 MiB it stops fitting an L3 alongside the working set, and
/// the per-task pack is ~1/(pixels per task) of the GEMM anyway.
constexpr int64_t kPrepackCap = int64_t{8} << 20;

/// Cap on the per-slot add_to residual / int32 accumulator tile set (elements).
/// The task's row chunks are split further until they fit, so a residual path
/// costs parallelism, never memory.
constexpr int64_t kTaskTileCap = int64_t{64} << 10;

// ---- tile helpers ---------------------------------------------------------

/// Seed an output tile [mc][nc] (row stride ld) with the per-column bias,
/// optionally adding what is already there (the identity-epilogue residual).
/// Without add_to the accumulator is built by the first MMA block and the bias
/// rides in the finish pass instead.
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
            for (int i = 0; i < nc; ++i) { row[i] = bias[i]; }
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
// HWIO hands the B packer its own order: k = (tap, input channel) and the
// output channel is contiguous, so one k row of a panel is one OC-length run of
// the weight — no transposing packer, no staging. A 1x1 filter's whole k slice
// is a single [icnc][ocnc] rectangle and packs in one call.
//
// With karea > 1 the taps are ic_per_group × OC apart, so the k slice is karea
// rectangles rather than one. Stage one panel's worth (nr_max OC values per k
// row) into the tap-major tile and pack from that: the staging copy is still
// contiguous per row, and the MMA keeps its one call over the whole kdim.
//
// When the weight fits the shared prepack all of this runs once per call. The
// staging is deliberately per *panel* rather than per block: nr_max × kdim is
// L1-sized (a few KB) instead of the whole block.
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

    // Input / output keep logical [N, C, H, W]; NHWC stores [N][H][W][C], so a
    // "row" is one spatial line (W pixels × C channels) and a pixel's channel
    // vector is contiguous.
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // HWIO: the output channel is the last axis, not the first.
    const int64_t OC = weight.shape(3);

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

    // A sample is IH rows of the channel-last pitch; a group is ic_per_group
    // channels inside every pixel.
    const int64_t ib_step = IH * in_row_stride;
    const int64_t ob_step = OH * out_row_stride;

    // HWIO (TensorFlow's filter): [KH][KW][IC/G][OC]. The output channel is the
    // contiguous axis, so one (tap, input channel) is an OC-length run and the B
    // panel is a plain strided copy of it — no transpose at all. Strides come
    // from the dims, not the pitch: a weight is a load-time constant, so a padded
    // weight is not a case this route has to absorb.
    //
    //   w_oc_stride  = 1                 (contiguous OC)
    //   w_ic_stride  = OC                (one input channel's row, all groups)
    //   w_tap_stride = ic_per_group * OC (one filter position)
    const int64_t w_row = weight.shape(3);                // [oc] run of one (tap, ic)
    const int64_t w_ic_stride = w_row;                    // [ic] within a tap
    const int64_t w_tap_stride = weight.shape(2) * w_row; // [kh][kw] position

    T* out_ptr = output.ptr<T>();
    const T* in_ptr = input.ptr<T>();
    const T* w_ptr  = weight.ptr<T>();
    const T* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    constexpr int64_t nr_max = nr_max_flt<T>();
    const int64_t esz = static_cast<int64_t>(sizeof(T));
    constexpr float kInf = std::numeric_limits<float>::infinity();

    // ---- the finish decision ----------------------------------------------
    // A pure clamp with no bias and no residual folds into the MMA store, the
    // same fold the planar route makes. Everything else needs a pass over the
    // finished tile: the bias is per *column* here (the accumulator's rows are
    // pixels), so the micro-kernel's per-row seed cannot carry it.
    float clamp_lo = -kInf;
    float clamp_hi = kInf;
    bool epi_folded = false;
    if (!attrs.add_to) {
        switch (attrs.epilogue.type) {
        case EpilogueActivateType::Relu:
            if (!has_bias) { clamp_lo = 0.0f; epi_folded = true; }
            break;
        case EpilogueActivateType::Relu6:
            if (!has_bias) { clamp_lo = 0.0f; clamp_hi = 6.0f; epi_folded = true; }
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

    // The seed is where the bias is added on the add_to route (it is per column,
    // so the micro-kernel cannot do it): the accumulator then starts at
    // `bias [+ residual]` and the epilogue must not add the bias a second time.
    // Everywhere else the MMA overwrites the tile and the bias rides in the
    // finish pass. With add_to the tile is always pre-loaded, so the MMA must
    // never zero it.
    const bool seed_bias = attrs.add_to && has_bias;
    const bool need_epi  = epi_pass || (has_bias && !attrs.add_to);

    // ---- block geometry ----------------------------------------------------
    const int64_t num_ic_blocks = (ic_per_group + icn_block - 1) / icn_block;
    const int64_t num_oc_blocks = (oc_per_group + ocn_block - 1) / ocn_block;
    const int64_t panels_per_oc = (ocn_block + nr_max - 1) / nr_max;
    const int64_t kdim_max = icn_block * karea;

    // A trailing ic block holds fewer channels, so its k is shorter and the MMA
    // expects a shorter panel stride. Both set sizes are known up front, which
    // is all the shared region's offsets need.
    const int64_t nfull_ic = ic_per_group / icn_block;
    const int64_t tail_ic  = ic_per_group - nfull_ic * icn_block;
    const int64_t set_full = gemm_set_elems(panels_per_oc, kdim_max, nr_max, esz, false);
    const int64_t set_tail = tail_ic
        ? gemm_set_elems(panels_per_oc, tail_ic * karea, nr_max, esz, false) : 0;
    const int64_t group_elems =
        nfull_ic * num_oc_blocks * set_full + (tail_ic ? num_oc_blocks * set_tail : 0);

    // ---- scratch -----------------------------------------------------------
    const Conv2DGemmScratch sc = gemm_scratch_layout(plan, esz);
    char* const base = static_cast<char*>(workspace);
    T* const shared_ptr = reinterpret_cast<T*>(base + sc.panel_off);
    const bool shared = plan.shared_elems > 0;

    // ---- the shared weight prepack (once per call) -------------------------
    if (shared) {
        const int64_t per_g = num_ic_blocks * num_oc_blocks;
        ctx.cpu.run(0, G * per_g, [&](int64_t s) {
            // Workers stage into their own slot's staging tile. The prepack
            // finishes before the first task starts, so the slot is free.
            char* const slot_ptr = base + sc.head +
                static_cast<int64_t>(ctx.cpu.current_thread_id()) * sc.slot_stride;
            T* const stage = reinterpret_cast<T*>(slot_ptr + sc.off_stage);

            const int64_t g    = s / per_g;
            const int64_t rem  = s % per_g;
            const int64_t icn  = (rem / num_oc_blocks) * icn_block;
            const int64_t ocn  = (rem % num_oc_blocks) * ocn_block;
            const int64_t icnc = std::min(icn_block, ic_per_group - icn);
            const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn);
            const int64_t kdim = icnc * karea;
            const bool    tail = icn >= nfull_ic * icn_block;
            const int64_t off  = tail
                ? g * group_elems + nfull_ic * num_oc_blocks * set_full
                      + (ocn / ocn_block) * set_tail
                : g * group_elems + (icn / icn_block) * num_oc_blocks * set_full
                      + (ocn / ocn_block) * set_full;
            pack_block<T>(shared_ptr + off, gemm_panel_slot(kdim, nr_max, esz, false),
                          stage,
                          w_ptr + g * oc_per_group + ocn + icn * w_ic_stride,
                          w_ic_stride, w_tap_stride, ocnc, icnc, karea,
                          nr_max, static_cast<int>(kdim));
        });
    }

    const int64_t NG = N * G;
    const int64_t chunks_per_ng = oh_chunks * oc_chunks;
    const int64_t total = NG * chunks_per_ng;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < total;

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

        // Contiguous, disjoint row range of this oh-chunk, and its oc range.
        const int64_t r0 = chunk * OH / oh_chunks;
        const int64_t r1 = (chunk + 1) * OH / oh_chunks;
        const int64_t oc_lo = oc_chunk * oc_per_group / oc_chunks;
        const int64_t oc_hi = (oc_chunk + 1) * oc_per_group / oc_chunks;

        // A pixel tile is one contiguous run of the output only when consecutive
        // rows are dense; then it may cross a row boundary, which is what keeps a
        // narrow plane's tiles at the full M instead of paying the micro-kernel's
        // row granularity per row (a 7-pixel row of a 6-row MR block wastes 40%).
        // The pointwise route reads A the same way, so it needs the input dense
        // too; the gather route addresses the input through the row pitch.
        const bool flat = out_row_stride == OW * OC &&
                          (!plan.pointwise || in_row_stride == IW * IC);

        for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block) {
            const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);
            // The residual region covers one pass's rows for this oc block (only
            // one oc block is live at a time).
            T* const orig = keep_orig ? orig_buf : nullptr;

            for (int64_t pr = r0; pr < r1; pr += ph_block) {
            const int64_t pr1 = std::min(r1, pr + ph_block);
            const int64_t p_first = pr * OW;
            const int64_t p_last  = pr1 * OW;
            // A pass of 49 pixels is otherwise 24 + 24 + 1, and the 1-row tile
            // pays a whole MR block for one row. Only a *short* tail is worth
            // re-splitting, and the new step stays a multiple of the micro-kernel's
            // row granularity — 24 + 24 + 12 costs exactly as many MR blocks as
            // three even 16s, and a step of 23 would pay a fourth block for 5 rows.
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
                if (shared) {
                    const bool is_tail = icn >= nfull_ic * icn_block;
                    const int64_t off = is_tail
                        ? g * group_elems + nfull_ic * num_oc_blocks * set_full
                              + (ocn / ocn_block) * set_tail
                        : g * group_elems + (icn / icn_block) * num_oc_blocks * set_full
                              + (ocn / ocn_block) * set_full;
                    b_panel = shared_ptr + off;
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
                                // The residual stays in the accumulator only on
                                // the identity-epilogue route; with keep_orig the
                                // saved copy carries it and the tile is the seed.
                                tile_seed<T>(tile, static_cast<int>(OC), mc,
                                             static_cast<int>(ocnc), bias_g + ocn,
                                             /*add=*/!keep_orig);
                            } else if (keep_orig) {
                                // No bias to seed with: the tile still holds the
                                // residual, which the epilogue adds back itself.
                                tile_zero<T>(tile, static_cast<int>(OC), mc,
                                             static_cast<int>(ocnc));
                            }
                        }

                        tile_mma_direct(mc, static_cast<int>(ocnc), kdim,
                                        tile, static_cast<int>(OC),
                                        a, lda, b_panel, c_lo, c_hi,
                                        /*zero_mode=*/icn == 0 && !attrs.add_to,
                                        nullptr);

                        if (is_last_icn && need_epi) {
                            epilogue_inplace<T>(mc, static_cast<int>(ocnc), tile,
                                                static_cast<int>(OC),
                                                (has_bias && !attrs.add_to)
                                                    ? bias_g + ocn : nullptr,
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
// The same product with integer operands: A is the packed activation tile
// (bytes, with the A shift applied), B the packed s8 HWIO weight, the
// accumulator int32 — one read-modify-write pass per ic block over the task's
// oc slice — and the epilogue compensates the zero points and either
// dequantizes or requantizes. An fp activation is materialized into a compact
// s8 plane once per call for the same reason the planar route does it.

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

/// Build the compact s8 activation plane the gather and the pointwise route
/// read: dense [N][H][W][IC], walked through the source's row pitch (a row is
/// one spatial line of W × C elements).
///
/// @p zero already carries the A shift: a u8 target enters zp − 128, so the
/// quantizer's own clamp lands the bytes in signed range and nothing has to XOR
/// the plane afterwards. A pre-quantized u8 activation is not quantized at all —
/// it is copied with that XOR — and an s8 one never reaches here (it is read in
/// place).
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

    // Below this a dispatch costs more than the pass (the pool re-spawns its
    // workers per run), the same split the planar plane quantizer makes.
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

/// Pack one (ic block, oc block) panel set from the HWIO weight. Mirrors the
/// float route's pack_block: the output channel is the contiguous axis, so a
/// k row of a panel is one OC-length run and nothing is transposed. A 1x1 filter
/// packs in one call; with karea > 1 the taps are rectangles of their own, so
/// one panel's rows are restaged tap-major first.
///
/// @p a_biased must be set whenever the kernel folds an A-side bias out of its
/// accumulators through the panel's column sums — on x86 always, since that is
/// where the +128 the VNNI kernel applies to every A group is cancelled.
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

    const int64_t OC = weight.shape(3);

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

    // HWIO (TensorFlow's filter): [KH][KW][IC/G][OC]. Dims, not pitch — the
    // contiguous axis is the output channel (dim 3), so one (tap, input channel)
    // is one OC-length run, same as in the fp kernel above.
    const int64_t w_row = weight.shape(3);
    const int64_t w_ic_stride = w_row;
    const int64_t w_tap_stride = weight.shape(2) * w_row;

    // ---- quantization parameters -----------------------------------------
    const QuantParams& qa = input.quant_params();
    const QuantParams& qw = weight.quant_params();

    const float   s_x  = quant_scale_at(qa, 0);
    const int32_t zp_x = quant_zp_at(qa, 0);
    NNOPS_ASSERT(s_x > 0.0f);

    // `d`: the shift the A bytes carry into the GEMM. Both arches' kernels fold
    // out the +128 *they* add and nothing else, so the accumulator holds
    // Σ stored·w and the epilogue must undo `d` itself. Which one applies is
    // decided by the *quantization target*: the activation's own storage type
    // when it arrives quantized, `act_quant_dtype` when it is quantized here —
    // the zero point means nothing without it.
    const bool u8a = kFpAct ? (attrs.act_quant_dtype == DataType::u8) : kU8;
    const int32_t d = u8a ? -128 : 0;

    // The compensation's two halves: the column term exists when the A side's
    // shift and zero point do not cancel (a per-tensor value here), the
    // per-pixel term when some weight zero point is non-zero.
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
        // s8 bytes, compact: the quantizer (or the pre-quantized copy) already
        // carries the shift. The zero point has to move into the *byte* domain
        // too — a u8 source's `s` is `q − 128`, i.e. `q + d` — or the quantizer
        // would clamp a u8-range zero against the s8 clamp.
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

    // R_b[oc] = Σ_k weight[oc,k] over the whole K — one SIMD pass over the same
    // bytes the panel pack streams, so it stays serial (the planar route makes
    // the same call for the same reason).
    // In HWIO one output channel's K values are a *strided* column, so the sum
    // accumulates over the k rows instead — elementwise across OC, which is the
    // contiguous axis and vectorises.
    std::vector<int32_t> r_b_all;
    if (col_term) {
        r_b_all.assign(static_cast<size_t>(OC), 0);
        for (int t = 0; t < karea; ++t) {
            for (int64_t i = 0; i < ic_per_group; ++i) {
                const int8_t* const row = w_ptr + t * w_tap_stride + i * w_ic_stride;
                for (int64_t oc = 0; oc < OC; ++oc) {
                    r_b_all[static_cast<size_t>(oc)] += row[oc];
                }
            }
        }
    }

    // ---- block geometry ----------------------------------------------------
    constexpr int64_t nr_max = NR_MAX_I8;
    const int64_t num_ic_blocks = (ic_per_group + icn_block - 1) / icn_block;
    const int64_t num_oc_blocks = (oc_per_group + ocn_block - 1) / ocn_block;
    const int64_t panels_per_oc = (ocn_block + nr_max - 1) / nr_max;
    const int64_t kdim_max = icn_block * karea;

    const int64_t nfull_ic = ic_per_group / icn_block;
    const int64_t tail_ic  = ic_per_group - nfull_ic * icn_block;
    const int64_t set_full = gemm_set_elems(panels_per_oc, kdim_max, nr_max, 1, true);
    const int64_t set_tail = tail_ic
        ? gemm_set_elems(panels_per_oc, tail_ic * karea, nr_max, 1, true) : 0;
    const int64_t group_elems =
        nfull_ic * num_oc_blocks * set_full + (tail_ic ? num_oc_blocks * set_tail : 0);

    const bool shared = plan.shared_elems > 0;
    const bool a_biased = arch::mma_i8_folds_a_bias;
    int8_t* const shared_ptr = reinterpret_cast<int8_t*>(base + sc.panel_off);

    // ---- the shared weight prepack (once per call) -------------------------
    if (shared) {
        const int64_t per_g = num_ic_blocks * num_oc_blocks;
        ctx.cpu.run(0, G * per_g, [&](int64_t s) {
            char* const slot_ptr = base + sc.head +
                static_cast<int64_t>(ctx.cpu.current_thread_id()) * sc.slot_stride;
            int8_t* const stage = reinterpret_cast<int8_t*>(slot_ptr + sc.off_stage);

            const int64_t g    = s / per_g;
            const int64_t rem  = s % per_g;
            const int64_t icn  = (rem / num_oc_blocks) * icn_block;
            const int64_t ocn  = (rem % num_oc_blocks) * ocn_block;
            const int64_t icnc = std::min(icn_block, ic_per_group - icn);
            const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn);
            const int64_t kdim = icnc * karea;
            const bool    tail = icn >= nfull_ic * icn_block;
            const int64_t off  = tail
                ? g * group_elems + nfull_ic * num_oc_blocks * set_full
                      + (ocn / ocn_block) * set_tail
                : g * group_elems + (icn / icn_block) * num_oc_blocks * set_full
                      + (ocn / ocn_block) * set_full;
            pack_block_i8(shared_ptr + off, gemm_panel_slot(kdim, nr_max, 1, true),
                          stage,
                          w_ptr + g * oc_per_group + ocn + icn * w_ic_stride,
                          w_ic_stride, w_tap_stride, ocnc, icnc, karea,
                          nr_max, static_cast<int>(kdim), a_biased);
        });
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
        const int64_t oc_lo = oc_chunk * oc_per_group / oc_chunks;
        const int64_t oc_hi = (oc_chunk + 1) * oc_per_group / oc_chunks;

        // Same flat-pixel rule as the fp route: a dense output row makes the
        // pixel tiles contiguous, so a tile may cross a row boundary and a
        // narrow plane stops paying the micro-kernel's row granularity per row.
        // The pointwise route reads A straight out of the input, so it needs
        // the input dense too.
        const bool flat = out_row_stride == OW * OC &&
                          (!plan.pointwise || in_row_stride == IW * IC);

        for (int64_t ocn = oc_lo; ocn < oc_hi; ocn += ocn_block) {
            const int64_t ocnc = std::min(ocn_block, oc_hi - ocn);

            for (int64_t pr = r0; pr < r1; pr += ph_block) {
            const int64_t pr1 = std::min(r1, pr + ph_block);
            const int64_t p_first = pr * OW;
            const int64_t p_last  = pr1 * OW;
            // A pass whose tail is shorter than the micro-kernel's row
            // granularity pays a whole MR block for a few pixels; re-split it
            // evenly, but keep the step a multiple of MR so the re-split costs
            // no extra block.
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
                // The pointwise route reads the pixels straight out of the row,
                // which only works when the block's k is a whole number of
                // 4-byte groups; otherwise the gather zero-fills the tail.
                const bool direct_a = plan.pointwise && (kdim == kdim4);

                const int8_t* b_panel;
                if (shared) {
                    const bool tail = icn >= nfull_ic * icn_block;
                    const int64_t off = tail
                        ? g * group_elems + nfull_ic * num_oc_blocks * set_full
                              + (ocn / ocn_block) * set_tail
                        : g * group_elems + (icn / icn_block) * num_oc_blocks * set_full
                              + (ocn / ocn_block) * set_full;
                    b_panel = shared_ptr + off;
                } else {
                    pack_block_i8(pack_buf, gemm_panel_slot(kdim, nr_max, 1, true), stage,
                                  w_g + ocn + icn * w_ic_stride,
                                  w_ic_stride, w_tap_stride, ocnc, icnc, karea, nr_max,
                                  kdim, a_biased);
                    b_panel = pack_buf;
                }

                // `mc` lives outside the loop: it is the step, and the body's
                // early exits (a non-final ic block, the epilogue) still have to
                // advance it.
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
                        // per k group), not row-major: its direct-A variant has to
                        // broadcast every group out of the row, which costs about
                        // as many uops as the dot products themselves. The pack is
                        // one pass over the tile the gather just wrote.
                        //
                        // The first K block stores its accumulators instead of
                        // adding them, so the tile needs no zero fill at all: it
                        // is a write of the whole int32 tile that only existed to
                        // be read back by the first MMA.
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
                        // whole K and of the *true* activation values, so the
                        // gather's shift comes back out. Recomputed rather than
                        // buffered — only a weight with a non-zero zero point
                        // needs it at all.
                        int32_t ra_tile[64];
                        const int32_t* ra_ptr = nullptr;
                        if (comp) {
                            NNOPS_ASSERT(mc <= 64);
                            if (direct_a) {
                                // The pixel is its own patch and its channels are
                                // one contiguous run, so the sum is a row sum of
                                // the same bytes the MMA reads — no gather, and no
                                // A tile (the plan reserves none on this route).
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
                        bool use_thread_slots)
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
        // The bias is typed by the branch: the output dtype on an fp destination,
        // int32 on an integer one.
        const DataType bias_dt = int_out ? DataType::s32 : out.dtype;
        if (inputs.size() > 2 && inputs[2].dtype != bias_dt) {
            return plan;
        }
        // add_to needs the pre-existing output inside the accumulator; only the
        // reference implements that fold on the quantized route.
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
    // The channel-last weight is TensorFlow's [KH, KW, IC/G, OC]; a planar one is
    // OIHW. Anything else is not this route's.
    if (!is_channel_last_weight(wt.layout)) { return plan; }
    const int64_t OC = wt.dims[3];
    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;
    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];

    // A 1x1 kernel at unit stride with no padding reads the input row itself as
    // A — no gather at all. Channel-last needs no pitch test for it: a pixel's
    // channels are contiguous whatever the row pitch, and a tile never leaves
    // its output row.
    plan.pointwise = (karea == 1 && attrs.stride[0] == 1 && attrs.stride[1] == 1 &&
                      attrs.padding[0] == 0 && attrs.padding[1] == 0);

    plan.icn_block = std::min<int64_t>(get_ic_block_nhwc(karea, ic_per_group),
                                       ic_per_group);
    // Balance the oc blocks instead of taking kOcnBlock and leaving a sliver: the
    // accumulator fill, the A pack and the epilogue are per oc block, so an
    // oc_per_group of 144 is 128 + 16 (an eighth-full block that pays all three)
    // rather than 80 + 64.
    const int64_t oc_blocks_wanted = (oc_per_group <= kOcnSingleCap)
        ? 1
        : (oc_per_group + kOcnBlock - 1) / kOcnBlock;
    const int64_t nr_for_split = quant ? NR_MAX_I8
                                       : ((dt == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32);
    const int64_t per_block = (oc_per_group + oc_blocks_wanted - 1) / oc_blocks_wanted;
    plan.ocn_block = std::min<int64_t>(
        oc_per_group,
        ((per_block + nr_for_split - 1) / nr_for_split) * nr_for_split);
    plan.m_block   = kMBlock;
    plan.oh_chunks = 1;
    plan.oc_chunks = 1;

    const int64_t NG = N * G;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    const int64_t esz = quant ? 1 : static_cast<int64_t>(data_type_size(dt));

    // Rows first (the axis every task shares), then the filter axis when the
    // rows cannot fill the pool — the same order and the same nearest-round rule
    // as the planar plan (an odd block count otherwise leaves a straggler wave
    // the barrier waits on).
    //
    // A row chunk still has to be worth a micro-kernel tile: split to a single
    // row of a 7x7 plane and every task is seven pixels against a 24-row tile —
    // a sixth of the MMA's rows busy, for two thirds of the time it takes the
    // same layer at 4 threads. So the row split stops at the number of chunks
    // that keeps `kTaskMinPixels` in each, and the filter axis takes the rest.
    const int64_t ow_at_least = std::max<int64_t>(OW, 1);
    const int64_t rows_min = std::max<int64_t>(1, (kTaskMinPixels + ow_at_least - 1) / ow_at_least);
    const int64_t max_row_chunks = std::max<int64_t>(1, OH / rows_min);
    plan.oh_chunks = std::clamp<int64_t>((nt + NG - 1) / NG, 1,
                                         std::min(std::max<int64_t>(OH, 1), max_row_chunks));
    const int64_t plane_tasks = NG * plan.oh_chunks;
    if (2 * plane_tasks < nt) {
        // The filter split divides N, so its granularity is a panel (`nr`), not
        // an M block: two panels per chunk is the smallest split that still
        // fills a micro-kernel tile. Splitting 96 oc seven ways leaves 14 live
        // lanes of a 16-wide panel — 21 tasks that all waste an eighth of N.
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

    // The residual tile set lives in the slot and must cover the task's whole row
    // range for one oc block. Split the rows further until it fits (parallelism,
    // not memory, is what a residual path spends).
    if (keep_orig) {
        while (plan.oh_chunks < OH &&
               ((OH + plan.oh_chunks - 1) / plan.oh_chunks) * OW * plan.ocn_block
                   > kTaskTileCap) {
            ++plan.oh_chunks;
        }
    }

    const int64_t rows_in_task = (OH + plan.oh_chunks - 1) / plan.oh_chunks;

    // Rows per accumulator pass. The accumulator (int32 on the quantized route,
    // the residual tile on add_to) is re-read and written once per ic block over
    // the rows one pass covers, so that set — not the whole task — is what has
    // to stay in cache: a 56×56 plane's full-tile set is 800 KB and every pass
    // went to DRAM (bench_conv2d_nhwc, where losing the epilogue alone
    // halved the time). Bound it and iterate the rows in chunks; the shared
    // prepack makes the extra passes free, and a task whose panels are packed
    // per task pays one pack per (oc block, pass, ic block) — a few hundred KB
    // of pixels per pack either way.
    const int64_t pass_esz = std::max<int64_t>(4, esz);
    plan.ph_block = std::clamp<int64_t>(
        kPassAccBytes / (pass_esz * plan.ocn_block * std::max<int64_t>(OW, 1)),
        1, std::max<int64_t>(rows_in_task, 1));

    const int64_t nr_max = quant ? NR_MAX_I8
                                 : ((dt == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32);
    const int64_t kdim_max = plan.icn_block * karea;
    const int64_t panels_per_oc = (plan.ocn_block + nr_max - 1) / nr_max;
    const int64_t num_ic_blocks = (ic_per_group + plan.icn_block - 1) / plan.icn_block;
    const int64_t num_oc_blocks = (oc_per_group + plan.ocn_block - 1) / plan.ocn_block;

    // Shared packed-B panels: one set per (group, ic block, oc block), written
    // once per call and read in place by every task. The trailing ic block packs
    // a shorter k and therefore a shorter panel stride, so it has its own size.
    const int64_t nfull_ic = ic_per_group / plan.icn_block;
    const int64_t tail_ic  = ic_per_group - nfull_ic * plan.icn_block;
    const int64_t set_full = gemm_set_elems(panels_per_oc, kdim_max, nr_max, esz, quant);
    const int64_t set_tail = tail_ic
        ? gemm_set_elems(panels_per_oc, tail_ic * karea, nr_max, esz, quant) : 0;
    const int64_t shared_elems =
        G * (nfull_ic * num_oc_blocks * set_full
             + (tail_ic ? num_oc_blocks * set_tail : 0));
    const bool shared = shared_elems * esz <= kPrepackCap;

    plan.shared_elems = shared ? shared_elems : 0;
    plan.pack_size = shared ? 0 : set_full;
    // int8 gathers bytes, padded to whole 4-byte groups (the direct-A kernel
    // reads kc four at a time) — which is also why a pointwise int8 tile whose
    // channel block is not a multiple of 4 still needs the gather buffer: the
    // in-row A read cannot be zero-padded.
    const bool pw_gather = quant && plan.pointwise && (ic_per_group % 4 != 0);
    // The quantized route hands the MMA a *packed* A panel (its direct-A kernel
    // broadcasts each k group from a row-major row, which costs ~2.7x the uops
    // of the packed one — see bench_conv2d_nhwc). The panel region comes
    // first in the slot's A area, so its stride is the gather's offset.
    plan.a_pack_stride = quant
        ? gemm_align64(static_cast<int64_t>(num_panels_max(static_cast<int>(plan.m_block), MR_I8))
                       * geom_ldd_i8_a((kdim_max + 3) & ~int64_t{3}))
        : 0;
    plan.a_size = plan.a_pack_stride
                + ((plan.pointwise && !pw_gather)
                       ? 0
                       : plan.m_block * (quant ? ((kdim_max + 3) & ~int64_t{3})
                                               : kdim_max + kABSlack));
    // In HWIO the output channel is the contiguous axis, so a panel's k row is
    // one OC-length run and nothing is transposed. Only a multi-tap filter needs
    // the staging tile: its taps are separate rectangles (ic_per_group × OC
    // apart), and one panel's rows are restaged tap-major before the pack.
    plan.stage_size = (karea > 1) ? nr_max * kdim_max : 0;
    const int64_t pass_tiles = plan.ph_block * OW;
    plan.orig_size = keep_orig ? pass_tiles * plan.ocn_block : 0;
    // The quantized route accumulates in int32 (its destination is s8/u8/fp), so
    // its accumulator is a per-slot tile set of the same shape as the residual.
    plan.acc_size = quant ? pass_tiles * plan.ocn_block : 0;
    // An fp activation is materialized into a compact s8 NHWC plane once per
    // call and read as bytes from then on — the same reason the planar route
    // does it: quantizing inside the gather would repeat it once per oc block.
    // A pre-quantized u8 one is materialized too when it takes the pointwise
    // route, whose in-row A read cannot apply the per-byte shift.
    const bool fp_act = (dt == DataType::f32 || dt == DataType::f16);
    plan.quant_size = (quant && (fp_act || (dt == DataType::u8 && plan.pointwise)))
                    ? N * in.dims[2] * in.dims[3] * IC : 0;

    const int64_t total = NG * plan.oh_chunks * plan.oc_chunks;
    int64_t slots = (use_thread_slots && nt < total) ? nt : total;
    // The shared prepack indexes its staging tile by worker id, so the slot
    // region must cover the pool even when the task count does not.
    if (plan.stage_size > 0) { slots = std::max(slots, nt); }
    plan.num_slots = slots;

    plan.workspace_size = static_cast<size_t>(gemm_scratch_layout(plan, esz).total);
    return plan;
}


// ---- dispatch -------------------------------------------------------------

/// The quantized route's dtype dispatch. Same shape as the planar one; the plan
/// already refused every unsupported combination.
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

/// The channel-last driver: quantized weights first (they own both the activation
/// dispatch and the epilogue choice), then the two fp element types.
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

/// A route is the pair the entry below needs: resolve a plan from the
/// descriptors, then run it. The two are template parameters to
/// conv2d_gemm_run so the plan/pool/dispatch prologue is written once.
struct PlanarRoute {
    static Conv2DGemmPlan make_plan(const Conv2DAttributes& attrs,
                                    std::span<const TensorDesc> inputs,
                                    std::span<const TensorDesc> outputs,
                                    int num_threads, bool use_thread_slots)
    {
        return planar::make_plan(attrs, inputs, outputs, num_threads, use_thread_slots);
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
                                    int num_threads, bool use_thread_slots)
    {
        return channel_last::make_plan(attrs, inputs, outputs, num_threads,
                                       use_thread_slots);
    }
    static Status run(const Conv2DAttributes& attrs, TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx, const Conv2DGemmPlan& plan,
                      void* workspace)
    {
        return channel_last::run(attrs, output, inputs, ctx, plan, workspace);
    }
};

/// The prologue every route shares: descriptors from the views, one plan, one
/// pooled arena sized by that plan, then the route's own driver. A zeroed plan
/// (icn_block 0) is the "unsupported dtype combination" answer — the same one
/// the route drivers give, refused before anything is allocated.
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
        attrs, descs, outs, ctx.cpu.thread_count(), ctx.cpu.thread_slot_count() > 0);
    if (plan.icn_block <= 0) { return Status::unsupported; }

    // A pointwise call with no add_to needs no scratch at all; the pool rejects
    // a zero-byte request, so ask for the smallest arena instead.
    PoolPtr scratch(plan.workspace_size ? plan.workspace_size : 1);
    if (!scratch) { return Status::out_of_memory; }

    return TRoute::run(attrs, output, inputs, ctx, plan, scratch.get());
}

}  // anonymous namespace

Conv2DGemmPlan get_conv2d_implicit_gemm_plan(const Conv2DAttributes& attrs,
                                             std::span<const TensorDesc> inputs,
                                             std::span<const TensorDesc> outputs,
                                             int num_threads,
                                             bool use_thread_slots)
{
    if (is_channel_last(inputs[0].layout)) {
        return ChannelLastRoute::make_plan(attrs, inputs, outputs, num_threads,
                                           use_thread_slots);
    }
    return PlanarRoute::make_plan(attrs, inputs, outputs, num_threads,
                                  use_thread_slots);
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
