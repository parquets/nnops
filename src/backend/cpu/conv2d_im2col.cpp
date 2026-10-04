/// @file conv2d_im2col.cpp
/// @brief Tiled im2col + direct GEMM Conv2D kernel (planar NCHW, f32 / f16).
///
/// Algorithm (ported from nn_compute im2col_gemm_conv2d_impl):
///   For each (batch, group, oh-chunk) — parallel via ctx.cpu.run:
///     For each oh-block in the chunk's contiguous range of the output plane:
///       For each ic-block:
///         im2col the input block into a scratch matrix [K=icnc*karea][N=ohc*OW]
///         For each oc-block:
///           init output tile = bias (+ original output when add_to)
///           tile += weight[ocnc, K] × col_data[K, roi_area]  (tile_mma_direct)
///           (after the last ic-block) apply the epilogue in-place
///
/// See conv2d_im2col.h for the work split, the scratch layout, the bias /
/// epilogue / add_to semantics and the shared plan.

#include "conv2d_im2col.h"
#include "matmul_helper.h"                 // tile_mma_direct + arch panel constants
#include "simd_kernel/simd_epilogue.hpp"   // epilogue_inplace
#include "simd_kernel/simd_im2col.hpp"     // tiled_im2col_2d / tiled_im2col_3d
#include "common/memory_pool.hpp"          // internal scratch-memory pool
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace nnops::backend::cpu {

// Forward-declare the reference kernel for the fallback paths.
namespace reference {
void conv2d_ref(const Conv2DAttributes& attrs,
                TensorView& output,
                std::span<const TensorView> inputs,
                const ComputeContext& ctx,
                void* workspace);
}

namespace {

using namespace nnops::simd;
namespace k = nnops::kernel;  // pure SIMD kernels: tiled_im2col_2d / tiled_im2col_3d

// ---- block-size heuristics (ported from nn_compute) ----------------------

/// Input-channel block size from the kernel area. Larger kernels reuse fewer
/// input channels per im2col buffer to bound the scratch footprint.
inline int get_ic_block(int karea) noexcept {
    if (karea == 1)  { return 64; }
    if (karea <= 9)  { return 8;  }
    if (karea <= 25) { return 4;  }
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
void conv2d_im2col_impl(const Conv2DAttributes& attrs,
                        TensorView& output,
                        std::span<const TensorView> inputs,
                        const ComputeContext& ctx,
                        const Conv2DPlan& plan,
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
    // truth — conv2d_im2col_kernel resolved them once, with the sizing).
    const int64_t icn_block = plan.icn_block;
    const int64_t ocn_block = plan.ocn_block;
    const int64_t oh_block  = plan.oh_block;
    const int64_t num_oh_blocks = plan.num_oh_blocks;
    const int64_t oh_chunks     = plan.oh_chunks;

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
    const int64_t col_size  = plan.col_size;
    const int64_t orig_size = plan.orig_size;

    const int64_t NG = N * G;
    const int64_t total = NG * oh_chunks;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < total;

    T* col_base  = static_cast<T*>(workspace);
    T* orig_base = attrs.add_to ? col_base + plan.num_slots * col_size : nullptr;

    const bool epilogue_active = attrs.epilogue.type != EpilogueActivateType::None;

    const auto run_task = [&](int64_t t) {
        const int64_t ng    = t / oh_chunks;
        const int64_t chunk = t % oh_chunks;
        const int64_t n = ng / G;
        const int64_t g = ng % G;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;
        T* col_data = col_base + slot * col_size;
        T* orig = orig_base ? orig_base + slot * orig_size : nullptr;

        T* output_ptr = out_ptr + n * ob_step + g * og_step;
        const T* input_ptr  = in_ptr + n * ib_step + g * ig_step;
        const T* weight_ptr = w_ptr + g * oc_per_group * ic_per_group * karea;
        const T* bias_ptr   = b_ptr ? b_ptr + g * oc_per_group : nullptr;

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

            T* tile_base = output_ptr + oh * out_row_stride;

            for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                const bool is_last_icn = (icn + icn_block) >= ic_per_group;

                const T* input_local = input_ptr + icn * icn_step;
                k::tiled_im2col_2d<T>(col_data, input_local,
                                      0, static_cast<int>(oh),
                                      static_cast<int>(OW), static_cast<int>(ohc),
                                      roi_pad_h, roi_pad_w,
                                      static_cast<int>(PH), static_cast<int>(PW),
                                      static_cast<int>(icnc), static_cast<int>(IH), static_cast<int>(IW),
                                      static_cast<int>(icn_step), static_cast<int>(in_row_stride),
                                      static_cast<int>(KH), static_cast<int>(KW),
                                      static_cast<int>(SH), static_cast<int>(SW),
                                      static_cast<int>(DH), static_cast<int>(DW));

                for (int64_t ocn = 0; ocn < oc_per_group; ocn += ocn_block) {
                    const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn);
                    T* tile = tile_base + ocn * ocn_step;

                    if (icn == 0) {
                        // add_to + non-identity epilogue needs the original
                        // output preserved (epilogue(bias + Σ) then += orig).
                        if (attrs.add_to && epilogue_active) {
                            tile_save<T>(orig, tile, static_cast<int>(ocn_step),
                                         static_cast<int>(ocnc), static_cast<int>(roi_area));
                        }
                        tile_set_bias<T>(tile, static_cast<int>(ocn_step),
                                         static_cast<int>(ocnc), static_cast<int>(roi_area),
                                         bias_ptr ? bias_ptr + ocn : nullptr,
                                         attrs.add_to && !epilogue_active);
                    }

                    const T* a = weight_ptr + icn * karea + ocn * w_ocn_step;
                    // The tile was just initialised with the bias (or the saved
                    // original output) — every icn block must accumulate.
                    tile_mma_direct(static_cast<int>(ocnc), static_cast<int>(roi_area),
                                    static_cast<int>(icnc * karea),
                                    tile, static_cast<int>(ocn_step),
                                    a, static_cast<int>(w_ocn_step),
                                    col_data, static_cast<int>(roi_area),
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    /*zero_mode=*/false);

                    if (is_last_icn) {
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

}  // anonymous namespace

// =========================================================================
//  Plan — single source of truth for block sizes, parallelism + workspace
// =========================================================================

Conv2DPlan get_conv2d_plan(const Conv2DAttributes& attrs,
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

    Conv2DPlan plan;

    const DataType dt = in.dtype;
    if (dt != DataType::f32 && dt != DataType::f16) {
        return plan;
    }
    if (wt.dtype != dt || out.dtype != dt) {
        return plan;
    }
    if (inputs.size() > 2 && inputs[2].dtype != dt) {
        return plan;
    }

    const int64_t N  = in.dims[0];
    const int64_t IC = in.dims[1];
    const int64_t OC = wt.dims[0];
    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

    const int64_t karea = attrs.kernel_size[0] * attrs.kernel_size[1];
    const int64_t PH    = attrs.padding[0];

    plan.icn_block = std::min<int64_t>(get_ic_block(static_cast<int>(karea)), ic_per_group);
    plan.ocn_block = std::min<int64_t>(kOcnBlock, oc_per_group);
    plan.oh_block  = ((PH + 1 + kAlign - 1) / kAlign) * kAlign;

    plan.num_oh_blocks = (OH + plan.oh_block - 1) / plan.oh_block;

    // Split the oh dimension when (batch, group) alone under-subscribes the
    // thread pool (the common batch=1 / groups=1 case would otherwise run
    // single-threaded). Each oh-chunk owns its own im2col scratch, so cap the
    // chunk count at the number of oh-blocks.
    const int64_t NG = N * G;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    plan.oh_chunks = std::clamp<int64_t>((nt + NG - 1) / NG, 1, plan.num_oh_blocks);

    plan.col_size  = plan.icn_block * karea * plan.oh_block * OW;
    plan.orig_size = attrs.add_to ? plan.ocn_block * plan.oh_block * OW : 0;

    // Scratch slots: one per worker thread when the backend reports thread ids and
    // that is fewer than the task count (thread_id-indexed, reused across every
    // task a thread claims); else one per parallel task.
    const int64_t total = NG * plan.oh_chunks;
    plan.num_slots = (use_thread_slots && nt < total) ? nt : total;
    plan.workspace_size = static_cast<size_t>(plan.num_slots)
                        * static_cast<size_t>(plan.col_size + plan.orig_size)
                        * data_type_size(dt);
    return plan;
}

// =========================================================================
//  Public API
// =========================================================================

void conv2d_im2col_kernel(const Conv2DAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          void* /*workspace*/)
{
    const DataType dt = inputs[0].data_type();

    const bool dtypes_ok = (dt == DataType::f32 || dt == DataType::f16) &&
                           inputs[1].data_type() == dt &&
                           output.data_type() == dt &&
                           (inputs.size() <= 2 || inputs[2].data_type() == dt);

    if (dtypes_ok) {
        // Block sizes + scratch (im2col buffer) come from one plan, so the
        // kernel and its sizing agree by construction. Scratch is pooled
        // internally (getWorkspaceSize returns 0); sized and allocated once,
        // before the (batch, group, oh-chunk) parallel dispatch.
        std::vector<TensorDesc> descs;
        descs.reserve(inputs.size());
        for (const auto& t : inputs) { descs.push_back(t.desc()); }
        const TensorDesc outs[] = {output.desc()};
        const Conv2DPlan plan = get_conv2d_plan(attrs, descs, outs, ctx.cpu.thread_count(),
                                                ctx.cpu.thread_slot_count() > 0);
        PoolPtr scratch(plan.workspace_size);

        if (dt == DataType::f32) {
            conv2d_im2col_impl<float>(attrs, output, inputs, ctx, plan, scratch.as<float>());
        } else {
            conv2d_im2col_impl<half>(attrs, output, inputs, ctx, plan, scratch.as<half>());
        }
        return;
    }

    // Mismatched dtypes: the f32 reference is the correctness baseline (no
    // scratch). f16 has no reference, so a dtype mismatch there is a hard error.
    if (dt == DataType::f32) {
        reference::conv2d_ref(attrs, output, inputs, ctx, nullptr);
        return;
    }

    NNOPS_ASSERT(!"conv2d_im2col_kernel: f16 Conv2D dtype mismatch "
                   "(f16 input requires an f16 weight and f16 output)");
}

}  // namespace nnops::backend::cpu
