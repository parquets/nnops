/// @file conv2d_im2col_int8.cpp
/// @brief Quantized Conv2D kernel — s8 weight × u8/s8/f16/f32 activation →
///        f32 / f16 / s8 / u8 output (planar NCHW).
///
/// The floating-point route (conv2d_im2col.cpp) maps Conv2D onto
///   C[OC][roi_area] = W[OC][IC/G · karea] × col[IC/G · karea][roi_area]
/// and this one reuses that mapping verbatim with integer operands:
///
///   A = the weight  (M = oc_per_group, lds = ic_per_group·karea) — s8,
///       per-channel QuantParams;
///   B = the im2col activation (N = roi_area, K = ic_per_group·karea) — u8 or
///       s8 bytes.
///
/// Everything after that is the shipped int8 GEMM: pack A and B into the arch's
/// panels, accumulate int32, then the epilogue (simd_epilogue.hpp) compensates
/// the zero points and either dequantizes, adds the bias, clips and activates
/// (fp destination — dq_epilogue_tile), or adds an int32 bias, applies Relu and
/// requantizes with the output's QuantParams (s8/u8 destination —
/// int8_epilogue_tile, the same one MatMul's integer output uses).
///
/// An fp activation is quantized into a compact plane once, before the tile
/// loop (quantize_plane), and is bytes from then on — the same representation a
/// pre-quantized activation arrives in. Quantizing inside the im2col gather was
/// tried first and measured worse: the gather repeats once per oc-block and once
/// per oh-block, so the conversion cost multiplies while a single linear pass
/// does not, and only a materialized plane lets a 1x1 convolution take the
/// pointwise route (bench_conv2d conv2d_i8_fpact_vs_fp_sweep has both).
///
/// The pass is one linear read+write, ~10 us per 200k elements here. Planes up
/// to 2^19 elements stay on the calling thread, because a second `ctx.cpu.run`
/// costs ~100 us at 4 threads with a pool that re-spawns its workers per run;
/// larger ones split over the pool, where the pass is the bigger cost.
///
/// Quantizing per task instead — each task materializing only the input rows it
/// reads into its own slot's plane — was measured and rejected. It folds the
/// conversion into the split that already exists, worth ~3% at 4 threads, but
/// the halo rows are then converted again at every oh-chunk boundary: on the
/// single-image shapes that dominate, the slots sum to 1.1-1.4x the scratch of
/// one shared plane (bench_conv2d conv2d_i8_fpact_vs_fp_sweep prints both).
///
/// # The u8 activation
///
/// The kernels multiply signed bytes (VNNI u8×s8 on x86 after the A-side XOR,
/// SDOT s8×s8 on aarch64), so a u8 activation is shifted into the signed range
/// by the B packer (`src_shift`, XOR 0x80, i.e. value − 128). Three places must
/// agree on that shift:
///
///   * the packer's XOR, and the panel correction block that is summed *after*
///     it (x86 folds the A bias per panel);
///   * `zp_eff = zp_x − 128`, the epilogue's B zero point;
///   * `r_b -= 128·kdim` per ic-block, since the pack sums are unsigned.
///
/// # Compensation
///
/// The weight's A-side bias is `int8_a_pack_bias(s8)`: 128 on x86 (s8 → u8 for
/// VNNI) and 0 on aarch64 (SDOT takes s8 directly), and
/// `int8_epilogue_offset(s8) == 0` on both, so the accumulator holds the raw
/// s8·s8 dot product and matmul_int8_compensate applies unchanged. When every
/// zero point is already the compensated one (the usual symmetric W8A8 case),
/// the identity is detected and the r_a / r_b sweeps are skipped entirely.
///
/// # Accumulator range
///
/// The accumulator is int32, like MatMul's: |Σ a·b| ≤ K·128·128, which stays
/// under INT32_MAX until K ≈ 131k — the same bound the int8 MatMul route has.

#include "conv2d_im2col.h"
#include "backend/cpu/matmul/matmul_helper.h"                 // int8 pack / MMA, panel strides, arch bias
#include "backend/cpu/simd_kernel/simd_epilogue.hpp"   // dq/int8_epilogue_tile, quant_scale_at
#include "backend/cpu/simd_kernel/simd_im2col.hpp"     // tiled_im2col_2d
#include "backend/cpu/simd_kernel/simd_quant.hpp"      // quant_kernel::quantization (fp → s8/u8)
#include "backend/cpu/simd_kernel/simd_int8_reduce.hpp"// int8_row_sum / int8_col_sum_slab
#include "backend/cpu/common/memory_pool.hpp"          // internal scratch-memory pool
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace nnops::backend::cpu {

namespace {

namespace k = nnops::kernel;  // pure SIMD kernels: im2col, int8 reductions

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
void conv2d_int8_impl(const Conv2DAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      const Conv2DPlan& plan,
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
    // pack_b follow it per slot. acc and pack_a are addressed by their own
    // absolute byte offsets, which already account for the plane and for their
    // 64-byte alignment.
    char* base = static_cast<char*>(workspace) + plan.quant_size;
    TDst* col_base     = reinterpret_cast<TDst*>(base);
    // Packed B is always s8 bytes, whatever the activation storage is.
    int8_t* pack_b_base = reinterpret_cast<int8_t*>(base + plan.num_slots * plan.col_size);
    int32_t* acc_base  = reinterpret_cast<int32_t*>(static_cast<char*>(workspace) + plan.acc_offset);
    int8_t* pack_a_base = reinterpret_cast<int8_t*>(static_cast<char*>(workspace) + plan.pack_a_offset);

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

        TDst* col_data = col_base + slot * plan.col_size;
        int8_t* pack_b = pack_b_base + slot * plan.pack_size;
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
void conv2d_int8_act(const Conv2DAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     const Conv2DPlan& plan,
                     void* workspace)
{
    constexpr bool kPreQuant =
        std::is_same_v<TAct, int8_t> || std::is_same_v<TAct, uint8_t>;
    if constexpr (kPreQuant) {
        // A pre-quantized activation is already the storage type.
        conv2d_int8_impl<TAct, TAct, TOut>(attrs, output, inputs, ctx, plan, workspace);
    } else if (attrs.act_quant_dtype == DataType::u8) {
        conv2d_int8_impl<TAct, uint8_t, TOut>(attrs, output, inputs, ctx, plan, workspace);
    } else {
        conv2d_int8_impl<TAct, int8_t, TOut>(attrs, output, inputs, ctx, plan, workspace);
    }
}

/// Second-level dispatch: the output dtype.
template <class TAct>
void conv2d_int8_out(const Conv2DAttributes& attrs,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     const Conv2DPlan& plan,
                     void* workspace)
{
    switch (output.data_type()) {
    case DataType::f16:
        conv2d_int8_act<TAct, half>(attrs, output, inputs, ctx, plan, workspace);
        break;
    case DataType::s8:
        conv2d_int8_act<TAct, int8_t>(attrs, output, inputs, ctx, plan, workspace);
        break;
    case DataType::u8:
        conv2d_int8_act<TAct, uint8_t>(attrs, output, inputs, ctx, plan, workspace);
        break;
    default:
        conv2d_int8_act<TAct, float>(attrs, output, inputs, ctx, plan, workspace);
        break;
    }
}

}  // anonymous namespace

void conv2d_int8_kernel(const Conv2DAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* /*workspace*/)
{
    std::vector<TensorDesc> descs;
    descs.reserve(inputs.size());
    for (const auto& t : inputs) { descs.push_back(t.desc()); }
    const TensorDesc outs[] = {output.desc()};

    const Conv2DPlan plan = get_conv2d_plan(attrs, descs, outs, ctx.cpu.thread_count(),
                                            ctx.cpu.thread_slot_count() > 0);

    // A zeroed plan means an unknown dtype combination, or `add_to`, which the
    // int8 kernel does not fold into its accumulator.
    NNOPS_CHECK_MSG(plan.icn_block > 0,
                    "Conv2D: unsupported dtype combination, or add_to on the "
                    "quantized route");

    PoolPtr scratch(plan.workspace_size);
    void* ws = scratch.get();

    switch (inputs[0].data_type()) {
    case DataType::f32: conv2d_int8_out<float>(attrs, output, inputs, ctx, plan, ws); break;
    case DataType::f16: conv2d_int8_out<half>(attrs, output, inputs, ctx, plan, ws); break;
    case DataType::u8:  conv2d_int8_out<uint8_t>(attrs, output, inputs, ctx, plan, ws); break;
    default:
        NNOPS_ASSERT(inputs[0].data_type() == DataType::s8);
        conv2d_int8_out<int8_t>(attrs, output, inputs, ctx, plan, ws);
        break;
    }
}

}  // namespace nnops::backend::cpu
