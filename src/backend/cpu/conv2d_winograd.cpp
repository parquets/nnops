/// @file conv2d_winograd.cpp
/// @brief Winograd F(2×2, 3×3) Conv2D kernel (k3s1, planar NCHW, f32/f16).
///
/// See conv2d_winograd.h for the algorithm, the two-phase structure, the
/// scratch layout and the numeric-ordering rules.

#include "conv2d_winograd.h"
#include "matmul_helper.h"                  // tile_mma_direct
#include "simd_kernel/simd_winograd.hpp"    // BᵀdB / AᵀMA / GgGᵀ transforms
#include "common/memory_pool.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace nnops::backend::cpu {

namespace {

using namespace nnops::simd;
namespace k = nnops::kernel;

constexpr int kWinoTile = 2;  // F(2,3): 2×2 output, 4×4 input window
constexpr int kWinoAlpha = 4; // the 4×4 transform domain
constexpr int kWinoPos = kWinoAlpha * kWinoAlpha;  // 16 batched GEMMs

/// Fill a halo row: dst[0..n_cols) = src[valid_beg-PW .. valid_end-PW), zero
/// elsewhere. @p src_row points at the source element for dst[valid_beg].
template <class T>
inline void halo_row(T* NNOPS_RESTRICT dst, const T* NNOPS_RESTRICT src_row,
                     int n_cols, int valid_beg, int valid_end) noexcept {
    constexpr int L = simd_lane_for<T>;
    int i = 0;
    for (; i + L <= valid_beg; i += L) { v_store(dst + i, v_zero(dst + i)); }
    for (; i < valid_beg; ++i) { s_store(dst + i, 0.0f); }

    const int len = valid_end - valid_beg;
    int j = 0;
    for (; j + L <= len; j += L) { v_store(dst + valid_beg + j, v_load(src_row + j)); }
    for (; j < len; ++j) { s_store(dst + valid_beg + j, s_load(src_row + j)); }

    for (int t = valid_end; t < n_cols; ++t) { s_store(dst + t, 0.0f); }
}

/// Materialize one input channel's padded halo. halo[ph][pw] is the input value
/// at (ph − PH, pw − PW); out-of-range is zero. Row stride is @p halo_ld.
template <class T>
void build_halo(T* NNOPS_RESTRICT halo, int halo_ld,
                const T* NNOPS_RESTRICT in, int in_ld,
                int in_h, int in_w, int halo_h, int n_cols,
                int PH, int PW) noexcept {
    const int valid_beg = std::max(0, PW);
    const int valid_end = std::min(n_cols, PW + in_w);
    for (int ph = 0; ph < halo_h; ++ph) {
        T* dst = halo + static_cast<ptrdiff_t>(ph) * halo_ld;
        const int src_h = ph - PH;
        if (src_h < 0 || src_h >= in_h || valid_beg >= valid_end) {
            for (int t = 0; t < n_cols; ++t) { s_store(dst + t, 0.0f); }
            continue;
        }
        const T* src_row = in + static_cast<ptrdiff_t>(src_h) * in_ld
                             + (valid_beg - PW);
        halo_row<T>(dst, src_row, n_cols, valid_beg, valid_end);
    }
}

// ---- main kernel ----------------------------------------------------------

template <class T>
void conv2d_winograd_impl(const Conv2DAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const Conv2DWinogradPlan& plan,
                          void* workspace)
{
    const auto& input = inputs[0];
    const auto& u_tensor = inputs[1];  // prepacked transform-domain weight
    const bool has_bias = inputs.size() > 2;

    // Input: [N, IC, IH, IW]
    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // U: [16, OC, IC/G] — OC is the *total* output-channel count.
    const int64_t OC  = u_tensor.shape(1);
    const int64_t ICG = u_tensor.shape(2);
    const int64_t u_pitch = u_tensor.row_stride_elems();
    NNOPS_ASSERT(u_pitch == ICG);

    // Output: [N, OC, OH, OW]
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row_stride = output.row_stride_elems();
    NNOPS_ASSERT(output.shape(1) == OC);

    const int64_t G = attrs.groups;
    const int64_t ic_per_group = ICG;
    const int64_t oc_per_group = OC / G;

    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    const int64_t TH      = plan.num_tile_rows;
    const int64_t TW_pad  = plan.num_tile_cols_pad;
    const int64_t num_tiles = plan.num_tiles;

    const int64_t icn_block  = plan.icn_block;
    const int64_t ocn_block  = plan.ocn_block;
    const int64_t tile_block = plan.tile_block;
    const int64_t num_tile_blocks = plan.num_tile_blocks;
    const int64_t num_oc_blocks   = plan.num_oc_blocks;

    T* out_ptr = output.ptr<T>();
    const T* in_ptr = input.ptr<T>();
    const T* u_ptr  = u_tensor.ptr<T>();
    const T* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    // Element strides.
    const int64_t icn_step = IH * in_row_stride;       // between input channels
    const int64_t ib_step  = IC * icn_step;            // between batch samples
    const int64_t ig_step  = ic_per_group * icn_step;  // between groups (input)
    const int64_t ocn_step = OH * out_row_stride;      // between output channels
    const int64_t ob_step  = OC * ocn_step;            // between batch samples
    const int64_t og_step  = oc_per_group * ocn_step;  // between groups (output)

    const int64_t u_plane_step = OC * u_pitch;   // between transform positions
    const int64_t v_p_step     = ICG * num_tiles;  // between transform positions

    // Scratch: [halo][V][M × num_slots]. V and the halo are per-(batch, group)
    // and shared by both phases; M is per worker slot.
    T* halo_base = static_cast<T*>(workspace);
    T* v_all     = halo_base + plan.halo_size;
    T* m_all     = v_all + plan.v_size;

    const int64_t halo_ld = plan.halo_w;
    const int64_t halo_ch = plan.halo_h * plan.halo_w;

    const int64_t num_b_tasks = plan.num_b_tasks;
    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < num_b_tasks;

    const int64_t tw_groups = TW_pad / 8;

    // Current (batch, group) — set per iteration, read by the phase lambdas.
    const T* halo_src = nullptr;
    int64_t n = 0;
    int64_t g = 0;

    // Phase A: build the halo, then transform the input into V. V is fully
    // materialized here (the reference implementation's choice).
    const auto run_halo = [&](int64_t ic) {
        build_halo<T>(halo_base + ic * halo_ch, static_cast<int>(halo_ld),
                      halo_src + ic * icn_step, static_cast<int>(in_row_stride),
                      static_cast<int>(IH), static_cast<int>(IW),
                      static_cast<int>(plan.halo_h), static_cast<int>(plan.halo_w),
                      static_cast<int>(PH), static_cast<int>(PW));
    };

    const auto run_a = [&](int64_t t) {
        const int64_t th  = t / tw_groups;
        const int64_t tw0 = (t % tw_groups) * 8;
        // A tile's index is row-major over the padded grid (th·TW_pad + tw), so
        // each 8-tile group writes a slice offset by its tile row — and, because
        // the group never spans rows, always within one row.
        const int64_t row_off = th * TW_pad;
        for (int64_t ic = 0; ic < ic_per_group; ++ic) {
            k::winograd_input_tile8<T>(halo_base + ic * halo_ch, static_cast<int>(halo_ld),
                                       static_cast<int>(2 * th), static_cast<int>(2 * tw0),
                                       v_all + ic * num_tiles + row_off + tw0,
                                       static_cast<int>(v_p_step));
        }
    };

    // Phase B: 16 accumulating GEMMs, then the output transform + write-back.
    const auto run_b = [&](int64_t t) {
        int64_t idx = t;
        const int64_t tb = idx % num_tile_blocks; idx /= num_tile_blocks;
        const int64_t th = idx % TH;              idx /= TH;
        const int64_t ob = idx;

        const int64_t tw0 = tb * tile_block;
        const int64_t tw1 = std::min(tw0 + tile_block, TW_pad);
        const int64_t nc  = tw1 - tw0;

        const int64_t ocn0 = ob * ocn_block;
        const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn0);

        // This task's tiles start at the tile row's origin.
        const int64_t v_row_off = th * TW_pad;

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : t;
        T* m = m_all + slot * plan.m_size;

        // Accumulate over the input channels: M_p is zeroed on the first ic
        // block (zero_mode) and accumulated after. Bias is deliberately NOT
        // folded in — M is transform-domain, and Aᵀ(·)A would distort it.
        for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
            const int64_t icnc = std::min(icn_block, ic_per_group - icn);
            for (int p = 0; p < kWinoPos; ++p) {
                const T* a = u_ptr + p * u_plane_step
                                  + (g * oc_per_group + ocn0) * u_pitch + icn;
                const T* b = v_all + p * v_p_step + icn * num_tiles + v_row_off + tw0;
                T* c = m + p * (ocn_block * tile_block);
                tile_mma_direct(static_cast<int>(ocnc), static_cast<int>(nc),
                                static_cast<int>(icnc),
                                c, static_cast<int>(tile_block),
                                a, static_cast<int>(u_pitch),
                                b, static_cast<int>(num_tiles),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::infinity(),
                                /*zero_mode=*/icn == 0);
            }
        }

        // Output transform + bias + epilogue + write-back, per oc row.
        const int64_t oh0 = 2 * th;
        const bool row1_ok = (oh0 + 1) < OH;
        T* out_base = out_ptr + n * ob_step + g * og_step;

        for (int64_t oc = 0; oc < ocnc; ++oc) {
            const float bias_oc = b_ptr ? s_load(b_ptr + g * oc_per_group + ocn0 + oc) : 0.0f;
            T* row0 = out_base + (ocn0 + oc) * ocn_step + oh0 * out_row_stride;
            T* row1 = row1_ok ? row0 + out_row_stride : row0;
            for (int64_t tl = 0; tl < nc; tl += 8) {
                // ow0 is an absolute output column and row0/row1 point at column
                // 0 — so the tile offset lives in ow0 alone. Offsetting the
                // pointers by 2·tw as well would count the column twice and, on
                // any group past the first, write a full row out of place.
                k::winograd_output_tile8<T>(
                    m, static_cast<int>(ocn_block * tile_block),
                    static_cast<int>(tile_block), static_cast<int>(oc), static_cast<int>(tl),
                    row0, row1,
                    static_cast<int>(2 * (tw0 + tl)), static_cast<int>(OW), row1_ok,
                    attrs.epilogue, bias_oc, attrs.add_to);
            }
        }
    };

    for (n = 0; n < N; ++n) {
        const T* batch_in = in_ptr + n * ib_step;
        for (g = 0; g < G; ++g) {
            halo_src = batch_in + g * ig_step;
            ctx.cpu.run(0, ic_per_group, run_halo);
            ctx.cpu.run(0, plan.num_a_tasks, run_a);
            ctx.cpu.run(0, num_b_tasks, run_b);
        }
    }
}

}  // anonymous namespace

// =========================================================================
//  Plan
// =========================================================================

Conv2DWinogradPlan get_conv2d_winograd_plan(const Conv2DAttributes& attrs,
                                            std::span<const TensorDesc> inputs,
                                            std::span<const TensorDesc> outputs,
                                            int num_threads,
                                            bool use_thread_slots)
{
    Conv2DWinogradPlan plan;
    if (inputs.size() < 2 || outputs.empty()) { return plan; }

    const auto& in  = inputs[0];
    const auto& u   = inputs[1];
    const auto& out = outputs[0];

    const DataType dt = in.dtype;
    if (dt != DataType::f32 && dt != DataType::f16) { return plan; }
    if (u.dtype != dt || out.dtype != dt) { return plan; }
    if (inputs.size() > 2 && inputs[2].dtype != dt) { return plan; }

    // k3s1 only — the whole F(2,3) derivation assumes a 3×3, unit-stride,
    // undilated kernel.
    if (attrs.kernel_size[0] != 3 || attrs.kernel_size[1] != 3) { return plan; }
    if (attrs.stride[0] != 1 || attrs.stride[1] != 1) { return plan; }
    if (attrs.dilation[0] != 1 || attrs.dilation[1] != 1) { return plan; }
    if (in.rank != 4 || out.rank != 4 || u.rank != 3) { return plan; }
    if (u.dims[0] != kWinoPos) { return plan; }

    const int64_t IC = in.dims[1];
    const int64_t OC = u.dims[1];
    const int64_t ICG = u.dims[2];
    const int64_t G = attrs.groups;
    if (G < 1 || IC % G != 0 || OC % G != 0) { return plan; }
    if (ICG != IC / G) { return plan; }

    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];
    if (OH <= 0 || OW <= 0) { return plan; }

    const int64_t TH = (OH + 1) / 2;
    const int64_t TW = (OW + 1) / 2;
    const int64_t TW_pad = ((TW + 7) / 8) * 8;

    const int64_t oc_per_group = OC / G;

    plan.num_tile_rows     = TH;
    plan.num_tile_cols     = TW;
    plan.num_tile_cols_pad = TW_pad;
    plan.num_tiles         = TH * TW_pad;
    plan.halo_h            = 2 * TH + 2;
    plan.halo_w            = 2 * TW_pad + 2;

    // K is capped at the f16 accumulate budget; M at 16·ocn·tile elements so a
    // slot's M stays small; the tile block is 8 or 16 tiles (one or two vector
    // groups) to balance phase-B task count against M footprint.
    plan.icn_block  = std::min<int64_t>(ICG, 128);
    plan.ocn_block  = std::min<int64_t>(64, oc_per_group);
    plan.tile_block = std::clamp<int64_t>(TW_pad, 8, 16);

    plan.num_oc_blocks   = (oc_per_group + plan.ocn_block - 1) / plan.ocn_block;
    plan.num_tile_blocks = (TW_pad + plan.tile_block - 1) / plan.tile_block;
    plan.num_a_tasks     = TH * (TW_pad / 8);
    plan.num_b_tasks     = plan.num_oc_blocks * TH * plan.num_tile_blocks;

    plan.halo_size = ICG * plan.halo_h * plan.halo_w;
    plan.v_size    = static_cast<int64_t>(kWinoPos) * ICG * plan.num_tiles;
    plan.m_size    = static_cast<int64_t>(kWinoPos) * plan.ocn_block * plan.tile_block;

    const int64_t nt = std::max<int64_t>(num_threads, 1);
    plan.num_slots = (use_thread_slots && nt < plan.num_b_tasks) ? nt : plan.num_b_tasks;

    plan.workspace_size = static_cast<size_t>(plan.halo_size + plan.v_size
                                              + plan.num_slots * plan.m_size)
                        * data_type_size(dt);
    return plan;
}

// =========================================================================
//  Public API
// =========================================================================

void conv2d_winograd_kernel(const Conv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* /*workspace*/)
{
    // Never fall back to reference::conv2d_ref: U is rank-3 with dims[0] == 16
    // and the reference would read it as an OC of 16.
    NNOPS_ASSERT(inputs.size() >= 2);
    const DataType dt = inputs[0].data_type();
    const bool dtypes_ok = (dt == DataType::f32 || dt == DataType::f16) &&
                           inputs[1].data_type() == dt &&
                           output.data_type() == dt &&
                           (inputs.size() <= 2 || inputs[2].data_type() == dt);
    NNOPS_ASSERT(dtypes_ok && "conv2d_winograd_kernel: dtype mismatch");
    NNOPS_ASSERT(inputs[1].layout() == TensorLayout::PackedWinoWeight &&
                 "conv2d_winograd_kernel: inputs[1] must be prepacked U "
                 "(run prepackWeights first)");

    std::vector<TensorDesc> descs;
    descs.reserve(inputs.size());
    for (const auto& t : inputs) { descs.push_back(t.desc()); }
    const TensorDesc outs[] = {output.desc()};
    const Conv2DWinogradPlan plan =
        get_conv2d_winograd_plan(attrs, descs, outs, ctx.cpu.thread_count(),
                                 ctx.cpu.thread_slot_count() > 0);
    NNOPS_ASSERT(plan.workspace_size > 0 &&
                 "conv2d_winograd_kernel: unsupported conv geometry (k3s1 only)");

    PoolPtr scratch(plan.workspace_size);
    if (dt == DataType::f32) {
        conv2d_winograd_impl<float>(attrs, output, inputs, ctx, plan, scratch.as<float>());
    } else {
        conv2d_winograd_impl<half>(attrs, output, inputs, ctx, plan, scratch.as<half>());
    }
}

}  // namespace nnops::backend::cpu
