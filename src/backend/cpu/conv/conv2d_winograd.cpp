/// @file conv2d_winograd.cpp
/// @brief Winograd F(2×2, 3×3) Conv2D kernel (k3s1, f32/f16, NCHW or NHWC).
///
/// Both routes live here as `planar` and `channel_last`, in the same spirit as
/// Conv2D's implicit-GEMM file: one plan per route, one dispatch, and the same
/// numeric order and scratch conventions on each.
///
/// See conv2d_winograd.h for the algorithm, the structure of the task grids, the
/// scratch layouts and the numeric-ordering rules.

#include "conv2d_winograd.h"
#include "backend/cpu/matmul/matmul_helper.h"                  // tile_mma_direct / tile_pack_rhs
#include "backend/cpu/simd_kernel/simd_winograd.hpp"    // BᵀdB / AᵀMA / GgGᵀ transforms
#include "backend/cpu/common/memory_pool.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

namespace nnops::backend::cpu {

namespace k = nnops::kernel;

constexpr int kWinoAlpha = 4; // the 4×4 transform domain (F(2,3): 2×2 output)
constexpr int kWinoPos = kWinoAlpha * kWinoAlpha;  // 16 batched GEMMs

// =========================================================================
//  Planar route (NCHW)
// =========================================================================

namespace planar {
namespace {

using namespace nnops::simd;

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

/// Materialize one input channel's padded-halo *strip*: row r of @p dst is padded
/// halo row (row0 + r), column c is padded column (col0 + c) — i.e. the input
/// value at (row0 + r − PH, col0 + c − PW), zero outside the input. Row stride is
/// @p ld. The coordinates are in the *global* padded grid, so a task can build
/// just the rows and columns its tiles read and index them locally.
template <class T>
void build_halo_strip(T* NNOPS_RESTRICT dst, int ld,
                      const T* NNOPS_RESTRICT in, int in_ld,
                      int in_h, int in_w,
                      int row0, int col0, int rows, int n_cols,
                      int PH, int PW) noexcept {
    // Padded column c is inside the input iff PW <= c < PW + in_w.
    const int valid_beg = std::max(col0, PW) - col0;
    const int valid_end = std::min(col0 + n_cols, PW + in_w) - col0;
    for (int r = 0; r < rows; ++r) {
        T* d = dst + static_cast<ptrdiff_t>(r) * ld;
        const int src_h = row0 + r - PH;
        if (src_h < 0 || src_h >= in_h || valid_beg >= valid_end) {
            for (int t = 0; t < n_cols; ++t) { s_store(d + t, 0.0f); }
            continue;
        }
        const T* src_row = in + static_cast<ptrdiff_t>(src_h) * in_ld
                             + (col0 + valid_beg - PW);
        halo_row<T>(d, src_row, n_cols, valid_beg, valid_end);
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

    const int64_t icn_block  = plan.icn_block;
    const int64_t ocn_block  = plan.ocn_block;
    const int64_t tile_block = plan.tile_block;
    const int64_t num_tile_blocks = plan.num_tile_blocks;
    const int64_t oc_chunks       = plan.oc_chunks;

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

    T* slot_base = static_cast<T*>(workspace);

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int esz    = static_cast<int>(sizeof(T));

    // One slot's scratch: [halo strip][V][M][packed-B]. Every task owns the whole
    // chain for its own tiles — halo rows and columns, the transformed V, the
    // transform-domain accumulator M and the packed panels — so tasks share no
    // scratch and need no ordering between them.
    const int64_t halo_ld = plan.halo_ld;              // 2·tile_block + 2 columns
    const int64_t halo_ch = kWinoAlpha * halo_ld;      // one channel's strip
    const int64_t v_p_step = plan.v_p_stride;          // one transform position's V plane

    const int64_t num_tasks = plan.num_tasks;
    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && plan.num_slots < num_tasks;

    // One task = one (batch, group, tile row, tile block, oc-chunk): build the
    // halo strip its tiles read, transform it, run the 16 GEMMs for its own
    // slice of the filter, and write those tiles back. Tile rows, blocks and oc
    // ranges are disjoint, so the whole convolution is one dispatch with no
    // phase between them.
    const auto run_task = [&](int64_t t) {
        int64_t idx = t;
        // The oc-chunk varies fastest: the chunks of one plane are adjacent in
        // task order, so the pool hands them out together. They rebuild the same
        // halo strip, and whichever arrives second finds those input lines in
        // L2 — ordering them apart would turn a shared read into n_planes reads.
        const int64_t oc_chunk = idx % oc_chunks; idx /= oc_chunks;
        const int64_t tb = idx % num_tile_blocks; idx /= num_tile_blocks;
        const int64_t th = idx % TH;              idx /= TH;
        const int64_t g  = idx % G;               idx /= G;
        const int64_t n  = idx;

        const int64_t tw0 = tb * tile_block;
        const int64_t nc  = std::min(tile_block, TW_pad - tw0);

        // This task's filter rows, also disjoint. An even split of the rows (not
        // of the oc-block list): the task count is what the pool wants, and an
        // even row split keeps the tasks within one row of each other — a block
        // split leaves the last task with two blocks while its neighbours run
        // one, and the barrier is set by the slowest.
        const int64_t oc_lo = oc_chunk * oc_per_group / oc_chunks;
        const int64_t oc_hi = (oc_chunk + 1) * oc_per_group / oc_chunks;

        const int64_t slot = per_thread
            ? static_cast<int64_t>(ctx.cpu.current_thread_id())
            : std::min(t, plan.num_slots - 1);
        T* scratch = slot_base + slot * plan.slot_size;
        T* halo    = scratch;
        T* v_all   = halo + plan.halo_size;
        T* m_all_  = v_all + plan.v_size;
        T* pack_b  = m_all_ + plan.m_size;

        // Halo strip: padded rows [2·th, 2·th + 4) — the four window rows of this
        // tile row — and halo_ld = 2·tile_block + 2 columns starting at the
        // block's first tile. Neighbouring tasks repeat the two window columns
        // their tile blocks share; that copy is cheaper than a phase boundary.
        const T* task_in = in_ptr + n * ib_step + g * ig_step;
        for (int64_t ic = 0; ic < ic_per_group; ++ic) {
            build_halo_strip<T>(halo + ic * halo_ch, static_cast<int>(halo_ld),
                                task_in + ic * icn_step, static_cast<int>(in_row_stride),
                                static_cast<int>(IH), static_cast<int>(IW),
                                static_cast<int>(2 * th), static_cast<int>(2 * tw0),
                                kWinoAlpha,
                                static_cast<int>(halo_ld),
                                static_cast<int>(PH), static_cast<int>(PW));
        }

        // Input transform of this block's tiles, for every input channel.
        for (int64_t ic = 0; ic < ic_per_group; ++ic) {
            T* v = v_all + ic * tile_block;
            const T* h = halo + ic * halo_ch;
            for (int64_t tg = 0; tg < nc; tg += 8) {
                k::winograd_input_tile8<T>(h, static_cast<int>(halo_ld), 0,
                                           static_cast<int>(2 * tg), v + tg,
                                           static_cast<int>(v_p_step));
            }
        }

        for (int64_t ocn0 = oc_lo; ocn0 < oc_hi; ocn0 += ocn_block) {
            const int64_t ocnc = std::min(ocn_block, oc_hi - ocn0);

            // Accumulate over the input channels: M_p is zeroed on the first ic
            // block (zero_mode) and accumulated after. Bias is deliberately NOT
            // folded in — M is transform-domain, and Aᵀ(·)A would distort it.
            for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                for (int p = 0; p < kWinoPos; ++p) {
                    const T* a = u_ptr + p * u_plane_step
                                      + (g * oc_per_group + ocn0) * u_pitch + icn;
                    const T* b = v_all + p * v_p_step + icn * tile_block;
                    T* c = m_all_ + p * (ocn_block * tile_block);

                    // Pack this transform position's V slice [icnc][nc] into
                    // uniform nr_max-wide panels; every oc row in the tile
                    // reuses it, so the copy costs 1/ocnc of the GEMM's MACs.
                    const int ldd_b = static_cast<int>(
                        mma_pack_align(nr_max * static_cast<int>(icnc) * esz) / esz);
                    tile_pack_rhs(false, static_cast<int>(nc), static_cast<int>(icnc),
                                  pack_b, ldd_b, b, static_cast<int>(tile_block), 1.0f);

                    tile_mma_direct(static_cast<int>(ocnc), static_cast<int>(nc),
                                    static_cast<int>(icnc),
                                    c, static_cast<int>(tile_block),
                                    a, static_cast<int>(u_pitch),
                                    pack_b,
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
                    // ow0 is an absolute output column and row0/row1 point at
                    // column 0 — so the tile offset lives in ow0 alone.
                    // Offsetting the pointers by 2·tw as well would count the
                    // column twice and, on any group past the first, write a
                    // full row out of place.
                    k::winograd_output_tile8<T>(
                        m_all_, static_cast<int>(ocn_block * tile_block),
                        static_cast<int>(tile_block), static_cast<int>(oc),
                        static_cast<int>(tl),
                        row0, row1,
                        static_cast<int>(2 * (tw0 + tl)), static_cast<int>(OW), row1_ok,
                        attrs.epilogue, bias_oc, attrs.add_to);
                }
            }
        }
    };

    ctx.cpu.run(0, num_tasks, run_task);
}

}  // anonymous namespace

/// Resolve the planar route: U is [16][OC][IC/G] and the activation is NCHW.
Conv2DWinogradPlan make_plan(const Conv2DAttributes& attrs,
                             std::span<const TensorDesc> inputs,
                             std::span<const TensorDesc> outputs,
                             int num_threads,
                             bool use_thread_slots,
                             int64_t num_slots)
{
    Conv2DWinogradPlan plan;
    plan.route = Conv2DWinogradRoute::Planar;
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

    // K is capped at the f16 accumulate budget; M at 16·ocn·tile elements so a
    // slot's M stays small; the tile block is 8 or 16 tiles (one or two vector
    // groups, and at least one full nr_max panel) to balance task count against
    // the V/M footprint a task carries.
    plan.icn_block  = std::min<int64_t>(ICG, 128);
    plan.ocn_block  = std::min<int64_t>(64, oc_per_group);
    plan.tile_block = std::clamp<int64_t>(TW_pad, 8, 16);

    plan.num_tile_blocks = (TW_pad + plan.tile_block - 1) / plan.tile_block;

    // One task per (batch, group, tile row, tile block). The batch and group
    // axes are part of the task grid, not a serial loop around one dispatch, so
    // N·G also counts towards filling the pool.
    const int64_t NB = (in.dims[0] > 0) ? in.dims[0] : 1;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    const int64_t plane_tasks = NB * G * TH * plan.num_tile_blocks;

    // Second parallel axis: the filter rows, exactly as Conv2D's im2col plan
    // grows one (conv2d_implicit_gemm.cpp). A 14x14 output plane is 7 tile rows and one
    // 8-wide tile block, so the grid is 7 tasks — and the tile axis has nothing
    // left to give, because TW_pad is already one vector wide. Measured on this
    // host (bench_conv2d conv2d_wino_f32_14x14_256ch, ResNet-18's layer3.x): 20
    // threads ran that layer FASTER at 4 threads than at 20 — 0.45 vs 0.59 ms —
    // because 7 tasks over 20 workers leaves each task alone on a core, the
    // spread over the slow E-cores decides the time, and im2col's 390 GFLOPS
    // pulls ahead of winograd's 198.
    //
    // What a chunk costs is the halo strip it rebuilds and the input transform
    // it repeats — the two phases the task-local scratch was fused to avoid
    // sharing. That is IC·4·halo_ld copies plus ~32 adds per tile per channel,
    // against a GEMM of 16·M·tiles·IC MACs: on the shape that matters here the
    // duplicate is under a percent of the task, so 3 chunks cost ~2% and buy
    // the whole pool. It is NOT free at the M floor below, which is where it
    // has to stop.
    //
    // The chunk count is the NEAREST whole number to nt/plane_tasks, not the
    // ceiling. Splitting past the pool buys nothing and costs a whole second
    // wave: 16 threads on this 7-task plane want 2 chunks (14 tasks, one wave),
    // and ceil would pick 3 (21 tasks, one straggler) — 0.35 vs 0.48 ms on the
    // case above. Below 1.5·plane_tasks the round lands on 1 and the plane keeps
    // the single dispatch it always had, which is what 8 threads want too.
    // The floor is two M-panels of work per task.
    plan.oc_chunks = 1;
    if (plane_tasks < nt) {
        const int64_t mr = (dt == DataType::f16) ? MR_MAX_F16 : MR_MAX_F32;
        const int64_t max_chunks = std::max<int64_t>(1, oc_per_group / (2 * mr));
        const int64_t want = (nt + plane_tasks / 2) / plane_tasks;
        plan.oc_chunks = std::clamp<int64_t>(want, 1, max_chunks);
    }
    plan.num_tasks  = plane_tasks * plan.oc_chunks;
    plan.halo_ld    = 2 * plan.tile_block + 2;   // padded columns per strip
    plan.halo_rows  = kWinoAlpha;                // the 4 window rows of one tile row

    plan.halo_size = ICG * plan.halo_rows * plan.halo_ld;
    // One extra vector of slack per transform position: the 16 positions are
    // written one vector at a time, and a plane stride that lands on a
    // power-of-two multiple collapses those stores onto a handful of L1 sets.
    // Measured on 1x64x56x56 f32 (interleaved A/B): 1.50 -> 1.32 ms at 1 thread,
    // 0.50 -> 0.47 at 4; the f16 cases lose about a percentage point at 1 thread
    // to the same effect, so the slack pays for itself on every case here.
    plan.v_p_stride = ICG * plan.tile_block + 8;
    plan.v_size     = static_cast<int64_t>(kWinoPos) * plan.v_p_stride;
    plan.m_size    = static_cast<int64_t>(kWinoPos) * plan.ocn_block * plan.tile_block;

    // Uniform packed-B scratch: one V slice [icn_block][tile_block] per slot.
    const int64_t nr_max = (dt == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32;
    const int64_t esz    = static_cast<int64_t>(data_type_size(dt));
    const int64_t ldd_b_max = mma_pack_align(
        static_cast<size_t>(nr_max * plan.icn_block * esz)) / esz;
    plan.b_size = ((plan.tile_block + nr_max - 1) / nr_max) * ldd_b_max;

    plan.slot_size = plan.halo_size + plan.v_size + plan.m_size + plan.b_size;

    // Slots: one per worker when the caller reports worker ids, else one per
    // task (the same convention Conv2D's im2col plan uses — without a worker id
    // there is no way to pick a slot that two concurrent tasks will not share).
    // The slot count follows the *workers*, not the reported thread count: an
    // over-subscribed grid states more tasks than there are workers, and a slot
    // here is a whole tile block, so following the grid would multiply the
    // arena for nothing.
    const int64_t avail = (num_slots > 0) ? num_slots : nt;
    plan.num_slots = (use_thread_slots && avail < plan.num_tasks) ? avail
                                                                  : plan.num_tasks;

    plan.workspace_size = static_cast<size_t>(plan.num_slots * plan.slot_size)
                        * data_type_size(dt);
    return plan;
}

}  // namespace planar

// =========================================================================
//  Channel-last route (NHWC)
// =========================================================================

WinoChannelLastU wino_channel_last_u(int64_t ic_per_group, int64_t oc_per_group,
                                     int64_t groups, DataType dt) noexcept {
    WinoChannelLastU u;
    if (ic_per_group <= 0 || oc_per_group <= 0 || groups <= 0) { return u; }
    if (dt != DataType::f32 && dt != DataType::f16) { return u; }

    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    // The MMA's panel width, and the k-block length it will be given. kc divides
    // IC/G exactly, so the panels need no zero-padded k tail; the ceiling is the
    // same f16 accumulate budget MatMul uses (KC_CAP_F16), with f32 allowed to
    // run twice as wide because its accumulator is exact enough for it.
    const int64_t nr  = (dt == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32;
    const int64_t cap = (dt == DataType::f16) ? 128 : 256;
    int64_t kc = std::min(ic_per_group, cap);
    while (ic_per_group % kc != 0) { --kc; }

    u.nr    = nr;
    u.kc    = kc;
    u.nb    = ic_per_group / kc;
    u.jb    = (oc_per_group + nr - 1) / nr;
    u.jbg   = groups * u.jb;
    u.ldd   = static_cast<int64_t>(
        mma_pack_align(static_cast<size_t>(nr * kc) * static_cast<size_t>(esz)) / esz);
    u.plane = u.nb * u.jbg * u.ldd;
    u.total = static_cast<int64_t>(kWinoPos) * u.plane;
    return u;
}

namespace channel_last {
namespace {

using namespace nnops::simd;

/// Tiles per task. The slot has to hold a task's whole V and M, so this is the
/// knob that decides whether they stay in L2: 16·T·(IC/G + ocn) elements, which
/// the shapes Winograd is worth using on (32-256 channels, oc blocks of <= 64)
/// keep near 256 KiB. 24 is also 4 × mr_max(6) on both arches, so a full task
/// leaves no partial A panel.
constexpr int64_t kTileBlock = 24;
/// Growth stops here: below it a task re-reads its whole U slice (16·kc·oc)
/// for fewer than 8 tiles of MACs, and the transform's fixed cost per task
/// starts to show.
constexpr int64_t kTileBlockMin = 8;
/// Output channels per GEMM (the N side), snapped down to whole panels. The
/// same trade as the planar route's M block: every extra oc block costs another
/// pass over the task's tiles (the accumulator traffic and one more B read).
constexpr int64_t kOcnBlock = 64;

/// One slot's scratch is V then M — no halo strip (a window is read straight
/// from the input, and the two columns neighbouring tiles share are already in
/// L1) and no packed-B block (U is prepacked at load and read in place).
template <class T>
void conv2d_winograd_nhwc_impl(const Conv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               const Conv2DWinogradPlan& plan,
                               void* workspace)
{
    const auto& input = inputs[0];
    const auto& u_tensor = inputs[1];  // prepacked transform-domain weight
    const bool has_bias = inputs.size() > 2;

    // Input / output keep the logical [N, C, H, W]; NHWC stores [N][H][W][C], so
    // a "row" is one spatial line and a pixel's channels are contiguous.
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t IC = input.shape(1);
    const int64_t in_row = input.row_stride_elems();

    const int64_t OC = output.shape(1);
    const int64_t OH = output.shape(2);
    const int64_t OW = output.shape(3);
    const int64_t out_row = output.row_stride_elems();

    const int64_t G  = attrs.groups;
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;
    const int64_t PH = attrs.padding[0];
    const int64_t PW = attrs.padding[1];

    // U: [16][nb][G·jb][ldd] — the GEMM's B panels, one set per transform
    // position. The tag is checked by the kernel; the dims must match the
    // geometry this plan derived, which is the contract prepack wrote them under.
    const WinoChannelLastU& ug = plan.u;
    NNOPS_ASSERT(u_tensor.shape(0) == kWinoPos && u_tensor.shape(1) == ug.nb &&
                 u_tensor.shape(2) == ug.jbg && u_tensor.shape(3) == ug.ldd);

    const int64_t TW = plan.num_tile_cols;
    const int64_t tiles = plan.num_tiles;

    const int64_t tile_block = plan.tile_block;
    const int64_t num_tile_blocks = plan.num_tile_blocks;
    const int64_t oc_chunks = plan.oc_chunks;
    const int64_t ocn_block = plan.ocn_block;
    const int64_t v_lda = plan.v_lda;
    const int64_t v_p_stride = plan.v_p_stride;
    const int64_t m_oc_stride = plan.m_oc_stride;
    const int64_t m_p_stride = tile_block * m_oc_stride;
    const int64_t oc_blocks = (oc_per_group + ocn_block - 1) / ocn_block;

    T* out_ptr = output.ptr<T>();
    const T* in_ptr = input.ptr<T>();
    const T* u_ptr  = u_tensor.ptr<T>();
    const T* b_ptr  = has_bias ? inputs[2].ptr<T>() : nullptr;

    const int64_t ib_step = IH * in_row;
    const int64_t ob_step = OH * out_row;

    // Both dtypes have 8-lane vectors, so a channel block is one vector.
    constexpr int64_t vec = simd_lane_for<T>;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && plan.num_slots < plan.num_tasks;

    // One task = one (batch, group, tile block, oc chunk): transform its tiles
    // into V once, run the 16 GEMMs for its oc blocks into M, write them back.
    // Tile blocks and oc ranges are disjoint and V/M are per slot, so the whole
    // convolution is a single dispatch with no phase between the passes.
    const auto run_task = [&](int64_t t) {
        int64_t idx = t;
        // The oc chunk varies fastest, as in the planar route: the chunks of one
        // plane are adjacent in task order, and they re-read the same input
        // lines and the same V — whichever arrives second finds them in L2.
        const int64_t chunk = idx % oc_chunks; idx /= oc_chunks;
        const int64_t tb = idx % num_tile_blocks; idx /= num_tile_blocks;
        const int64_t g  = idx % G;               idx /= G;
        const int64_t n  = idx;

        const int64_t tile0 = tb * tile_block;
        const int64_t ntl = std::min(tile_block, tiles - tile0);

        const int64_t slot = per_thread
            ? static_cast<int64_t>(ctx.cpu.current_thread_id())
            : std::min(t, plan.num_slots - 1);
        T* scratch = static_cast<T*>(workspace) + slot * plan.slot_size;
        T* v_all = scratch;
        T* m_all = v_all + plan.v_size;

        const T* task_in  = in_ptr + n * ib_step + g * ic_per_group;
        T* task_out = out_ptr + n * ob_step + g * oc_per_group;

        // Input transform, every channel of the group, tile by tile: a tile's
        // 16 windows are read once and swept across the channel blocks, so the
        // four input pixels of the window stay in L1.
        for (int64_t tl = 0; tl < ntl; ++tl) {
            const int64_t tt = tile0 + tl;
            const int64_t th = tt / TW;
            const int64_t tw = tt - th * TW;
            const int64_t ih0 = 2 * th - PH;
            const int64_t iw0 = 2 * tw - PW;
            for (int64_t ic = 0; ic < ic_per_group; ic += vec) {
                const int lanes = static_cast<int>(std::min<int64_t>(vec, ic_per_group - ic));
                // The block's channel offset goes into both pointers: V's row
                // channel and the activation's channel lane.
                k::winograd_input_nhwc<T>(task_in + ic, in_row, IC,
                                          static_cast<int>(ih0), static_cast<int>(iw0),
                                          static_cast<int>(IH), static_cast<int>(IW),
                                          v_all + tl * v_lda + ic, v_p_stride, lanes);
            }
        }

        // This chunk's oc blocks, disjoint. Splitting on block boundaries (not on
        // column ranges) keeps every start a whole panel into the prepacked U.
        const int64_t b0 = chunk * oc_blocks / oc_chunks;
        const int64_t b1 = (chunk + 1) * oc_blocks / oc_chunks;

        for (int64_t b = b0; b < b1; ++b) {
            const int64_t ocn0 = b * ocn_block;
            const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn0);

            // Accumulate over the input channels: M_p is zeroed on the first
            // k-block (zero_mode) and accumulated after. Bias is deliberately NOT
            // folded in — M is transform-domain, and Aᵀ(·)A would distort it.
            for (int64_t kb = 0; kb < ug.nb; ++kb) {
                const int64_t kc = ug.kc;
                const T* b_base = u_ptr + (kb * ug.jbg + g * ug.jb + ocn0 / ug.nr) * ug.ldd;
                const T* a_base = v_all + kb * kc;
                for (int p = 0; p < kWinoPos; ++p) {
                    tile_mma_direct(static_cast<int>(ntl), static_cast<int>(ocnc),
                                    static_cast<int>(kc),
                                    m_all + p * m_p_stride,
                                    static_cast<int>(m_oc_stride),
                                    a_base + p * v_p_stride, static_cast<int>(v_lda),
                                    b_base + p * ug.plane,
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    /*zero_mode=*/kb == 0);
                }
            }

            // Output transform + bias + epilogue + write-back, per tile: the
            // 2×2 pixels of a tile are adjacent in memory, an odd output extent
            // leaving the fourth one (or the second row) uncovered.
            for (int64_t tl = 0; tl < ntl; ++tl) {
                const int64_t tt = tile0 + tl;
                const int64_t th = tt / TW;
                const int64_t tw = tt - th * TW;
                const int64_t oh = 2 * th;
                const int64_t ow = 2 * tw;
                for (int64_t oc = 0; oc < ocnc; oc += vec) {
                    const int lanes = static_cast<int>(std::min<int64_t>(vec, ocnc - oc));
                    T* p00 = task_out + oh * out_row + ow * OC + ocn0 + oc;
                    T* p01 = (ow + 1 < OW) ? p00 + OC : nullptr;
                    T* p10 = (oh + 1 < OH) ? p00 + out_row : nullptr;
                    T* p11 = (p01 != nullptr && p10 != nullptr) ? p01 + out_row : nullptr;
                    k::winograd_output_nhwc<T>(
                        m_all + tl * m_oc_stride + oc, m_p_stride,
                        p00, p01, p10, p11,
                        attrs.epilogue,
                        b_ptr ? b_ptr + g * oc_per_group + ocn0 + oc : nullptr,
                        attrs.add_to, lanes);
                }
            }
        }
    };

    ctx.cpu.run(0, plan.num_tasks, run_task);
}

/// Resolve the channel-last route: the activation is NHWC and U is prepacked as
/// B panels (PackedWinoWeightLast).
Conv2DWinogradPlan make_plan(const Conv2DAttributes& attrs,
                             std::span<const TensorDesc> inputs,
                             std::span<const TensorDesc> outputs,
                             int num_threads,
                             bool use_thread_slots,
                             int64_t num_slots)
{
    Conv2DWinogradPlan plan;
    plan.route = Conv2DWinogradRoute::ChannelLast;
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
    if (in.rank != 4 || out.rank != 4 || u.rank != 4) { return plan; }
    if (!is_channel_last(in.layout) || !is_channel_last(out.layout)) { return plan; }
    if (u.layout != TensorLayout::PackedWinoWeightLast) { return plan; }

    const int64_t IC = in.dims[1];
    const int64_t OC = out.dims[1];
    const int64_t G  = attrs.groups;
    if (G < 1 || IC % G != 0 || OC % G != 0) { return plan; }
    const int64_t ic_per_group = IC / G;
    const int64_t oc_per_group = OC / G;

    // The panels' geometry is derived, not read: the same function ran at prepack
    // time, and matching it against the descriptor is what keeps the two ends of
    // that contract from drifting.
    const WinoChannelLastU ug = wino_channel_last_u(ic_per_group, oc_per_group, G, dt);
    if (ug.total == 0 || u.dims[0] != kWinoPos || u.dims[1] != ug.nb ||
        u.dims[2] != ug.jbg || u.dims[3] != ug.ldd) { return plan; }
    plan.u = ug;

    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];
    if (OH <= 0 || OW <= 0) { return plan; }

    const int64_t TH = (OH + 1) / 2;
    const int64_t TW = (OW + 1) / 2;
    const int64_t tiles = TH * TW;

    plan.num_tile_rows     = TH;
    plan.num_tile_cols     = TW;
    plan.num_tile_cols_pad = tiles;   // the flattened tile count
    plan.num_tiles         = tiles;
    plan.icn_block         = ug.kc;

    // Whole panels: the U's oc axis is padded to them, and a GEMM's N start must
    // land on one so its B pointer is a panel boundary.
    const int64_t nr = ug.nr;
    const int64_t ocn_raw = std::min<int64_t>(
        kOcnBlock, ((oc_per_group + nr - 1) / nr) * nr);
    plan.ocn_block = std::max<int64_t>(nr, (ocn_raw / nr) * nr);

    // Tile blocks first (they are free — flattening means no work is repeated),
    // then the oc axis, which re-reads the U slice and repeats nothing else.
    const int64_t NB = (in.dims[0] > 0) ? in.dims[0] : 1;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    int64_t ntb = (tiles + kTileBlock - 1) / kTileBlock;
    while (NB * G * ntb < nt && tiles / (ntb + 1) >= kTileBlockMin) { ++ntb; }
    plan.tile_block = (tiles + ntb - 1) / ntb;
    plan.num_tile_blocks = (tiles + plan.tile_block - 1) / plan.tile_block;

    const int64_t oc_blocks = (oc_per_group + plan.ocn_block - 1) / plan.ocn_block;
    const int64_t plane_tasks = NB * G * plan.num_tile_blocks;
    plan.oc_chunks = 1;
    if (plane_tasks < nt && oc_blocks > 1) {
        const int64_t want = (nt + plane_tasks / 2) / plane_tasks;
        plan.oc_chunks = std::clamp<int64_t>(want, 1, oc_blocks);
    }
    plan.num_tasks = plane_tasks * plan.oc_chunks;

    // V: one [tile][ic] plane per transform position, tiles contiguous inside a
    // position (that is the GEMM's A, so lda is a tile row). The row is padded to
    // the vector width so the transform's tail block can be written whole; the
    // +8 per position keeps the 16 planes off one L1 set (the planar route's
    // measurement applies unchanged).
    plan.v_lda      = ((ic_per_group + 7) / 8) * 8;
    plan.v_p_stride = plan.tile_block * plan.v_lda + 8;
    plan.v_size     = static_cast<int64_t>(kWinoPos) * plan.v_p_stride;
    // M: [tile][oc] per position, a whole panel per row so the MMA's widest read
    // of a row stays inside it.
    plan.m_oc_stride = plan.ocn_block;
    plan.m_size      = static_cast<int64_t>(kWinoPos) * plan.tile_block * plan.m_oc_stride;

    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    plan.halo_size = 0;   // no halo strip on this route
    plan.b_size    = 0;   // and no packed-B block: U is read in place
    // Round the slot so every slot base keeps the transform's 32-byte stores
    // aligned for f16 as well.
    plan.slot_size = ((plan.v_size + plan.m_size) + 15) / 16 * 16;
    // The slot count follows the *workers*, not the reported thread count: an
    // over-subscribed grid states more tasks than there are workers, and a slot
    // here is a whole tile block, so following the grid would multiply the
    // arena for nothing.
    const int64_t avail = (num_slots > 0) ? num_slots : nt;
    plan.num_slots = (use_thread_slots && avail < plan.num_tasks) ? avail
                                                                  : plan.num_tasks;
    plan.workspace_size = static_cast<size_t>(plan.num_slots * plan.slot_size) *
                          static_cast<size_t>(esz);
    return plan;
}

}  // namespace
}  // namespace channel_last

Conv2DWinogradPlan get_conv2d_winograd_plan(const Conv2DAttributes& attrs,
                                            std::span<const TensorDesc> inputs,
                                            std::span<const TensorDesc> outputs,
                                            int num_threads,
                                            bool use_thread_slots,
                                            int64_t num_slots)
{
    // The activation's layout picks the route. The two U forms are not
    // interchangeable — each is the other's GEMM with its operands swapped — so
    // a mismatched pair must resolve to a zeroed plan (and the kernel's assert)
    // rather than to a route that would index it wrongly.
    if (!inputs.empty() && is_channel_last(inputs[0].layout)) {
        return channel_last::make_plan(attrs, inputs, outputs, num_threads,
                                       use_thread_slots, num_slots);
    }
    return planar::make_plan(attrs, inputs, outputs, num_threads, use_thread_slots,
                             num_slots);
}

// =========================================================================
//  Public API
// =========================================================================

Status conv2d_winograd_kernel(const Conv2DAttributes& attrs,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            void* /*workspace*/)
{
    // Never route to conv2d_cpu here: U is rank-3 with dims[0] == 16
    // and the reference would read it as an OC of 16.
    NNOPS_ASSERT(inputs.size() >= 2);
    const DataType dt = inputs[0].data_type();
    const bool dtypes_ok = (dt == DataType::f32 || dt == DataType::f16) &&
                           inputs[1].data_type() == dt &&
                           output.data_type() == dt &&
                           (inputs.size() <= 2 || inputs[2].data_type() == dt);
    NNOPS_ASSERT(dtypes_ok && "conv2d_winograd_kernel: dtype mismatch");
    NNOPS_ASSERT(is_wino_weight(inputs[1].layout()) &&
                 "conv2d_winograd_kernel: inputs[1] must be prepacked U "
                 "(run prepackWeights first)");

    std::vector<TensorDesc> descs;
    descs.reserve(inputs.size());
    for (const auto& t : inputs) { descs.push_back(t.desc()); }
    const TensorDesc outs[] = {output.desc()};
    // Slot sizing: per worker when the caller reports worker ids; one slot when
    // there is no pool at all (a serial run reuses it task after task); else one
    // per task — the plan's fallback when parallelism exists but cannot be
    // indexed (see Conv2DWinogradPlan::num_slots).
    const bool serial = !ctx.cpu.parallel_for;
    const Conv2DWinogradPlan plan =
        get_conv2d_winograd_plan(attrs, descs, outs, ctx.cpu.thread_count(),
                                 serial || ctx.cpu.thread_slot_count() > 0,
                                 ctx.cpu.thread_slot_count());
    NNOPS_ASSERT(plan.workspace_size > 0 &&
                 "conv2d_winograd_kernel: unsupported conv geometry (k3s1 only)");

    PoolPtr scratch(plan.workspace_size);
    const bool nhwc = plan.route == Conv2DWinogradRoute::ChannelLast;
    if (dt == DataType::f32) {
        if (nhwc) {
            channel_last::conv2d_winograd_nhwc_impl<float>(
                attrs, output, inputs, ctx, plan, scratch.as<float>());
        } else {
            planar::conv2d_winograd_impl<float>(
                attrs, output, inputs, ctx, plan, scratch.as<float>());
        }
    } else if (nhwc) {
        channel_last::conv2d_winograd_nhwc_impl<half>(
            attrs, output, inputs, ctx, plan, scratch.as<half>());
    } else {
        planar::conv2d_winograd_impl<half>(
            attrs, output, inputs, ctx, plan, scratch.as<half>());
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
