#pragma once
/// @file conv2d_im2col.h
/// @brief Planar (NCHW) Conv2D — tiled im2col + direct GEMM kernel.
///
/// Ported from nn_compute (`src/cpu/im2col_conv2d_cpu.cpp` +
/// `kernel/conv2d/nchw/implicit_gemm_conv2d.hpp` +
/// `kernel/conv2d/im2col_2d.hpp`). The output spatial plane is split into
/// oh-blocks; for each block the input is gathered straight into the uniform
/// nr_max-wide packed-B panels the GEMM consumes (tiled_im2col_2d_packed —
/// the im2col and pack steps fused into one pass, no intermediate col matrix)
/// and the output tile is accumulated with the raw (unpacked) weight through
/// the same `tile_mma_direct` dispatch that MatMul uses (matmul_helper.h).
/// The quantized route keeps the two-step form: it gathers bytes into a col
/// buffer and packs it separately (conv2d_im2col_int8.cpp).
///
/// A 1x1 kernel with unit stride and no padding needs no im2col at all — at
/// karea == 1 the gather degenerates to a dense row copy — so that case (the
/// plan's `pointwise` flag, set only for a dense input) runs the GEMM straight
/// off the input tensor: B is the input plane with ldb = the channel stride, and
/// K is the whole ic_per_group in one pass. MLAS takes the same route
/// (convolve.cpp:1654, MlasConvAlgorithmGemmDirect, on K == InputChannels).
///
/// Work is split at (batch, group, oh-chunk) granularity: when (batch, group)
/// alone cannot fill the thread pool (the common batch=1 / groups=1 case), the
/// oh-blocks are partitioned into parallel chunks. Scratch is indexed by thread
/// id when the backend reports worker ids; otherwise each chunk owns its own
/// im2col scratch. Output tiles are disjoint, so the result is bit-identical to
/// serial execution.
///
/// Bias is fused into the accumulator init; the epilogue (activation) is
/// applied in-place after the last ic-block; `add_to` follows the reference
/// semantics (epilogue applied to bias + Σ, then added to the pre-existing
/// output), preserving the original tile in a scratch buffer when needed.
///
/// Block sizes, the parallel decomposition, and the scratch layout all come
/// from one plan (get_conv2d_plan), so the kernel and its sizing never drift.
///
/// Scratch layout (pooled internally; one slot per worker thread when thread
/// ids are reported, else one slot per task, reused across the task's oh-blocks):
///   [quant    : N·IC·IH·IW bytes — the fp activation, quantized once per
///               call and shared by every task]
///   [col_data : icn_block*karea*oh_block*out_w — quantized route only]
///   [orig     : ocn_block*oh_block*out_w   (add_to only)]
///   [pack_b   : ceil(oh_block*out_w/nr_max) * ldd_b]
/// then, on the quantized route only (fp has both regions sized 0):
///   [acc      : ocn_block*oh_block*out_w    int32]
///   [pack_a   : num_panels_max(ocn_block, MR_I8) * ldd_a_i8]
///
/// col_data is the im2col matrix [K=icn_block*karea][N=oh_block*out_w]. The fp
/// route never builds it (col_size 0) — the gather writes the packed panels
/// directly — and neither does `pointwise`, where the packer reads B straight
/// from the input. orig preserves the output tile for the exact add_to
/// semantics. pack_b holds B in the uniform nr_max-wide panel layout the
/// unified MMA kernels consume (matmul_helper.h); it is rebuilt per ic-block.
/// The quant plane is written once per call (on the calling thread, before the
/// task loop) and then read as if the activation had arrived quantized — that is
/// what lets an fp activation take the pointwise route and stops the gather from
/// re-quantizing per oc-block. It is one plane for the whole call, not one per
/// task: a per-task plane would have to duplicate the halo rows at every
/// oh-chunk boundary and so holds more, not less, scratch on the single-image
/// shapes that dominate (bench_conv2d conv2d_i8_fpact_vs_fp_sweep prints it).

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/ops/conv2d.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace nnops::backend::cpu {

/// Single source of truth for the im2col block sizes, the oh-chunk parallel
/// decomposition, and the scratch layout (get_conv2d_plan).
struct Conv2DPlan {
    int64_t icn_block = 0;         ///< input-channel block
    int64_t ocn_block = 0;         ///< output-channel block
    int64_t oh_block  = 0;         ///< output-row block
    int64_t num_oh_blocks = 1;     ///< ceil(OH / oh_block)
    int64_t oh_chunks     = 1;     ///< parallel oh-chunks per (batch, group)
    int64_t num_slots = 0;         ///< scratch slots (worker threads, else tasks)
    int64_t col_size  = 0;         ///< im2col scratch elements per slot (0 → pointwise,
                                   ///< or the fp route, whose gather is fused with
                                   ///< the pack; only the quantized route builds it)
    int64_t orig_size = 0;         ///< add_to scratch elements per slot
    int64_t pack_size = 0;         ///< uniform packed-B elements per slot

    // ---- quantized route only (0 on the floating-point route) --------------
    int64_t acc_size     = 0;      ///< int32 accumulator elements per slot
                                   ///< (ocn_block·oh_block·OW)
    int64_t pack_a_size  = 0;      ///< packed-weight bytes per slot
    int64_t quant_size   = 0;      ///< bytes of the shared quantized-activation
                                   ///< plane at the front of the arena
                                   ///< (N·IC·IH·IW, one byte each). 0 when the
                                   ///< activation already arrives quantized.
    int64_t acc_offset    = 0;     ///< byte offset of the accumulator region
    int64_t pack_a_offset = 0;     ///< byte offset of the packed-A region

    bool pointwise = false;        ///< k1s1p0 + dense input: the GEMM reads the input
                                   ///< tensor directly, so there is no im2col buffer
                                   ///< and icn_block spans all of ic_per_group
    size_t  workspace_size = 0;    ///< pooled scratch bytes. 0 means either "unsupported
                                   ///< dtype → reference fallback" or "nothing to pool"
                                   ///< (pointwise with no add_to); the kernel tells those
                                   ///< apart by dtype, never by this value.
};

/// Resolve the conv2d plan (block sizes, oh-chunk split, workspace bytes) from
/// the descriptors. Returns a zeroed plan for unsupported dtype combinations,
/// which routes the kernel to the reference. Supported:
///   * f32/f16 on every tensor (the floating-point route);
///   * the quantized route — an s8 weight, a u8/s8/f16/f32 activation, an
///     f32/f16/s8/u8 output and a matching bias (see is_quantized_conv2d:
///     the bias is int32 on an s8/u8 destination, the output dtype otherwise),
///     without add_to (which only the reference implements).
/// @p num_threads drives the oh-chunk split; @p use_thread_slots sizes the
/// scratch per worker thread (thread_id-indexed) instead of per task.
Conv2DPlan get_conv2d_plan(const Conv2DAttributes& attrs,
                           std::span<const TensorDesc> inputs,
                           std::span<const TensorDesc> outputs,
                           int num_threads,
                           bool use_thread_slots = false);

/// Tiled im2col + direct GEMM Conv2D kernel (f32 / f16, planar NCHW).
///
/// Falls back to the reference kernel for f32 when the dtypes do not match (the
/// reference is f32-only). Scratch is pooled internally (sized by
/// get_conv2d_plan); @p workspace is vestigial.
Status conv2d_im2col_kernel(const Conv2DAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* workspace);

/// Quantized Conv2D kernel (conv2d_im2col_int8.cpp): s8 weight (per-channel
/// QuantParams) × u8/s8/f16/f32 activation (per-tensor QuantParams) → f32/f16
/// output, through the int8 GEMM and a dequantizing epilogue.
///
/// Same tiling, parallel decomposition and scratch layout as the floating-point
/// kernel, plus an int32 accumulator and a packed-weight region. A f32/f16
/// activation is quantized into a dense plane once per call (plan.quant_size,
/// at the front of the arena) and read as bytes from then on, so it follows the
/// same route as a pre-quantized one — including pointwise. `add_to` and every
/// unsupported combination go to the reference. Called from
/// conv2d_im2col_kernel when the weight is s8.
Status conv2d_int8_kernel(const Conv2DAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        void* workspace);

}  // namespace nnops::backend::cpu
