/// @file conv2d_winograd.cpp
/// @brief Winograd F(2×2, 3×3) Conv2D kernel (k3s1, f32/f16, channel-first or
/// channel-last).
///
/// Both routes live here as `channel_first` and `channel_last`, in the same spirit as
/// Conv2D's implicit-GEMM file: one plan per route, one dispatch, and the same
/// numeric order and scratch conventions on each.
///
/// See conv2d_winograd.h for the algorithm, the structure of the task grids, the
/// scratch layouts and the numeric-ordering rules.

#include "conv2d_winograd.h"
#include "backend/cpu/matmul/matmul_helper.h"                  // tile_mma_direct / tile_pack_rhs
#include "backend/cpu/simd_kernel/simd_winograd.hpp"    // BᵀdB / AᵀMA / GgGᵀ transforms
#include "backend/cpu/simd_kernel/simd_winograd43.hpp"  // the F(4,3) transforms
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
//  Channel-first route (channel-first)
// =========================================================================

namespace channel_first {
namespace {

using namespace nnops::simd;

/// dst[0..n_cols) = src_row[0 .. valid_end−valid_beg) placed at dst[valid_beg],
/// zero outside that span; @p src_row points at the element for dst[valid_beg].
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

/// One input channel's padded-halo *strip*: row r is padded row (row0 + r) and
/// column c padded column (col0 + c), i.e. input (row0+r−PH, col0+c−PW), zero
/// outside, at row stride @p ld. Coordinates are in the global padded grid, so a
/// task builds only the rows and columns its tiles read and indexes them locally.
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

template <class T, WinoOrder O>
void conv2d_winograd_impl(const Conv2DAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const Conv2DWinogradPlan& plan,
                          void* workspace)
{
    constexpr int64_t kpos  = wino_pos(O);
    constexpr int64_t ktile = wino_tile(O);
    constexpr int64_t kalpha = wino_alpha(O);  // the window: one tile plus the taps

    const auto& input = inputs[0];
    const auto& u_tensor = inputs[1];  // prepacked transform-domain weight
    const bool has_bias = inputs.size() > 2;

    const int64_t N  = input.shape(0);
    const int64_t IC = input.shape(1);
    const int64_t IH = input.shape(2);
    const int64_t IW = input.shape(3);
    const int64_t in_row_stride = input.row_stride_elems();

    // U: [pos, OC, IC/G] — OC is the *total* output-channel count.
    const int64_t OC  = u_tensor.shape(1);
    const int64_t ICG = u_tensor.shape(2);
    const int64_t u_pitch = u_tensor.row_stride_elems();
    NNOPS_ASSERT(u_pitch == ICG);

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
    const int64_t icn_step = IH * in_row_stride;
    const int64_t ib_step  = IC * icn_step;
    const int64_t ig_step  = ic_per_group * icn_step;
    const int64_t ocn_step = OH * out_row_stride;
    const int64_t ob_step  = OC * ocn_step;
    const int64_t og_step  = oc_per_group * ocn_step;

    const int64_t u_plane_step = OC * u_pitch;

    T* slot_base = static_cast<T*>(workspace);

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int esz    = static_cast<int>(sizeof(T));

    // One slot's scratch: [halo strip][V][M][packed-B], the whole chain a task
    // needs, so no two tasks share scratch and no ordering is required.
    const int64_t halo_ld = plan.halo_ld;              // tile·tile_block + slack
    const int64_t halo_ch = plan.halo_rows * halo_ld;  // one channel's strip
    const int64_t v_p_step = plan.v_p_stride;          // one V position's plane

    const int64_t num_tasks = plan.num_tasks;
    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && plan.num_slots < num_tasks;

    // One task = one (batch, group, tile-row group, tile block, oc-chunk): build
    // the halo strip its tiles read, transform it, run the pos GEMMs for its own
    // slice of the filter, and write those tiles back. Rows, blocks and oc
    // ranges are disjoint, so the whole convolution is one dispatch.
    const int64_t row_group = plan.tile_rows;          // tile rows one task spans
    const int64_t th_tasks  = (TH + row_group - 1) / row_group;
    const auto run_task = [&](int64_t t) {
        int64_t idx = t;
        // The oc-chunk varies fastest so the chunks of one plane are adjacent in
        // task order: they rebuild the same halo strip, and whichever arrives
        // second finds those input lines in L2. Ordering them apart would turn a
        // shared read into n_planes reads.
        const int64_t oc_chunk = idx % oc_chunks; idx /= oc_chunks;
        const int64_t tb = idx % num_tile_blocks; idx /= num_tile_blocks;
        const int64_t thg = idx % th_tasks;       idx /= th_tasks;
        const int64_t g  = idx % G;               idx /= G;
        const int64_t n  = idx;

        const int64_t th0 = thg * row_group;
        const int64_t tw0 = tb * tile_block;
        // A task's live tiles: its tile-row group's columns, or one block of
        // columns when it owns a single row.
        const int64_t rows_here = std::min(row_group, TH - th0);
        const int64_t nc = (row_group > 1) ? rows_here * TW_pad
                                           : std::min(tile_block, TW_pad - tw0);

        // This task's filter rows, also disjoint, split evenly by row count
        // rather than by oc block: the slowest task sets the barrier, and a block
        // split can leave the last one with a block more than its neighbours.
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

        // Halo strip: the window rows this task's tile rows read, halo_ld
        // columns from the block's first tile. Neighbouring blocks repeat the
        // window columns they share; that copy is cheaper than a phase boundary.
        const T* task_in = in_ptr + n * ib_step + g * ig_step;
        for (int64_t ic = 0; ic < ic_per_group; ++ic) {
            build_halo_strip<T>(halo + ic * halo_ch, static_cast<int>(halo_ld),
                                task_in + ic * icn_step, static_cast<int>(in_row_stride),
                                static_cast<int>(IH), static_cast<int>(IW),
                                static_cast<int>(ktile * th0),
                                static_cast<int>(ktile * tw0),
                                static_cast<int>(plan.halo_rows),
                                static_cast<int>(halo_ld),
                                static_cast<int>(PH), static_cast<int>(PW));
        }

        // Input transform of this block's tiles, for every input channel. A group
        // of eight never leaves its tile row: the column count is a whole number
        // of them, and the destination tile index is the task's row-major one.
        // The columns are the *task's* — a single-row task owning one block of a
        // wide plane has nc of them, not TW_pad: the halo strip holds only this
        // block's tiles, and V is only tile_block wide.
        const int64_t cols = (row_group > 1) ? TW_pad : nc;
        for (int64_t ic = 0; ic < ic_per_group; ++ic) {
            T* v = v_all + ic * tile_block;
            const T* h = halo + ic * halo_ch;
            for (int64_t r = 0; r < rows_here; ++r) {
                const T* hr = h + r * ktile * halo_ld;
                T* vr = v + r * TW_pad;
                for (int64_t tg = 0; tg < cols; tg += 8) {
                    if constexpr (O == WinoOrder::F23) {
                        k::winograd_input_tile8<T>(hr, static_cast<int>(halo_ld), 0,
                                                   static_cast<int>(ktile * tg), vr + tg,
                                                   static_cast<int>(v_p_step));
                    } else {
                        k::winograd43_input_tile8<T>(hr, static_cast<int>(halo_ld), 0,
                                                     static_cast<int>(ktile * tg), vr + tg,
                                                     static_cast<int>(v_p_step));
                    }
                }
            }
        }

        for (int64_t ocn0 = oc_lo; ocn0 < oc_hi; ocn0 += ocn_block) {
            const int64_t ocnc = std::min(ocn_block, oc_hi - ocn0);

            // M_p zeroes on the first ic block and accumulates after; bias is
            // added at write-back, never here (Aᵀ(·)A would distort it).
            for (int64_t icn = 0; icn < ic_per_group; icn += icn_block) {
                const int64_t icnc = std::min(icn_block, ic_per_group - icn);
                for (int p = 0; p < kpos; ++p) {
                    const T* a = u_ptr + p * u_plane_step
                                      + (g * oc_per_group + ocn0) * u_pitch + icn;
                    const T* b = v_all + p * v_p_step + icn * tile_block;
                    T* c = m_all_ + p * (ocn_block * tile_block);

                    // Pack this position's V slice [icnc][nc] into nr_max-wide
                    // panels; every oc row of the tile reuses it, so the copy
                    // costs 1/ocnc of the GEMM's MACs.
                    const int ldd_b = static_cast<int>(
                        mma_pack_align(nr_max * static_cast<int>(icnc) * esz) / esz);
                    tile_pack_rhs(false, static_cast<int>(nc), static_cast<int>(icnc),
                                  pack_b, ldd_b, b, static_cast<int>(tile_block), 1.0f);

                    tile_mma_direct(static_cast<int>(ocnc), static_cast<int>(nc),
                                    static_cast<int>(icnc),
                                    c, static_cast<int>(tile_block),
                                    a, static_cast<int>(u_pitch),
                                    pack_b, /*ldb=*/-1, /*stage=*/nullptr,
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    /*zero_mode=*/icn == 0);
                }
            }

            // Output transform + bias + epilogue + write-back, per oc row and
            // per tile row of the task: a group of eight tiles is one row's.
            T* out_base = out_ptr + n * ob_step + g * og_step;

            for (int64_t oc = 0; oc < ocnc; ++oc) {
                const float bias_oc = b_ptr ? s_load(b_ptr + g * oc_per_group + ocn0 + oc) : 0.0f;
                for (int64_t r = 0; r < rows_here; ++r) {
                    const int64_t oh0 = ktile * (th0 + r);
                    if (oh0 >= OH) { break; }
                    const int rows_ok =
                        static_cast<int>(std::min<int64_t>(ktile, OH - oh0));
                    T* row0 = out_base + (ocn0 + oc) * ocn_step + oh0 * out_row_stride;
                    for (int64_t tl = 0; tl < nc; tl += 8) {
                        // ow0 is an absolute output column while the rows point
                        // at column 0, so the tile offset lives in ow0 alone;
                        // offsetting the pointers by tile·tw too would count the
                        // column twice (and write out of place once g > 0).
                        const int ow0 = static_cast<int>(ktile * (tw0 + tl));
                        const int t0 = static_cast<int>(r * TW_pad + tl);
                        if constexpr (O == WinoOrder::F23) {
                            const bool row1_ok = (oh0 + 1) < OH;
                            T* row1 = row1_ok ? row0 + out_row_stride : row0;
                            k::winograd_output_tile8<T>(
                                m_all_, static_cast<int>(ocn_block * tile_block),
                                static_cast<int>(tile_block), static_cast<int>(oc),
                                t0, row0, row1, ow0, static_cast<int>(OW), row1_ok,
                                attrs.epilogue, bias_oc, attrs.add_to);
                        } else {
                            T* rows[4];
                            for (int rr = 0; rr < 4; ++rr) {
                                rows[rr] = row0 + rr * out_row_stride;
                            }
                            k::winograd43_output_tile8<T>(
                                m_all_, static_cast<int>(ocn_block * tile_block),
                                static_cast<int>(tile_block), static_cast<int>(oc),
                                t0, rows, ow0, static_cast<int>(OW), rows_ok,
                                attrs.epilogue, bias_oc, attrs.add_to);
                        }
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, num_tasks, run_task);
}

}  // anonymous namespace

/// Resolve the channel-first route: U is [pos][OC][IC/G] and the activation
/// is channel-first.
Conv2DWinogradPlan make_plan(const Conv2DAttributes& attrs,
                             std::span<const TensorDesc> inputs,
                             std::span<const TensorDesc> outputs,
                             int num_threads,
                             bool use_thread_slots,
                             int64_t num_slots)
{
    Conv2DWinogradPlan plan;
    plan.route = Conv2DWinogradRoute::ChannelFirst;
    if (inputs.size() < 2 || outputs.empty()) { return plan; }

    const auto& in  = inputs[0];
    const auto& u   = inputs[1];
    const auto& out = outputs[0];

    const DataType dt = in.dtype;
    if (dt != DataType::f32 && dt != DataType::f16) { return plan; }
    if (u.dtype != dt || out.dtype != dt) { return plan; }
    if (inputs.size() > 2 && inputs[2].dtype != dt) { return plan; }

    // k3s1 only — both derivations assume a 3×3, unit-stride, undilated kernel.
    if (attrs.kernel_size[0] != 3 || attrs.kernel_size[1] != 3) { return plan; }
    if (attrs.stride[0] != 1 || attrs.stride[1] != 1) { return plan; }
    if (attrs.dilation[0] != 1 || attrs.dilation[1] != 1) { return plan; }
    if (in.rank != 4 || out.rank != 4 || u.rank != 3) { return plan; }

    // The transform order is the U's leading dimension, as on the channel-last
    // route: 16 positions is F(2,3), 36 is F(4,3).
    const WinoOrder order = (u.dims[0] == wino_pos(WinoOrder::F43)) ? WinoOrder::F43
                                                                   : WinoOrder::F23;
    const int64_t ktile = wino_tile(order);
    if (u.dims[0] != wino_pos(order)) { return plan; }
    plan.order = order;

    const int64_t IC = in.dims[1];
    const int64_t OC = u.dims[1];
    const int64_t ICG = u.dims[2];
    const int64_t G = attrs.groups;
    if (G < 1 || IC % G != 0 || OC % G != 0) { return plan; }
    if (ICG != IC / G) { return plan; }

    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];
    if (OH <= 0 || OW <= 0) { return plan; }

    const int64_t TH = (OH + ktile - 1) / ktile;
    const int64_t TW = (OW + ktile - 1) / ktile;
    const int64_t TW_pad = ((TW + 7) / 8) * 8;

    const int64_t oc_per_group = OC / G;

    plan.num_tile_rows     = TH;
    plan.num_tile_cols     = TW;
    plan.num_tile_cols_pad = TW_pad;
    plan.num_tiles         = TH * TW_pad;

    // K caps at 128 (the f16 accumulate budget, and applied to both dtypes), M at
    // 64 rows so a slot's M stays small; the tile block is 8 or 16 tiles — one or
    // two 8-wide vector groups — balancing task count against the V/M footprint
    // a task carries.
    plan.icn_block  = std::min<int64_t>(ICG, 128);
    plan.ocn_block  = std::min<int64_t>(64, oc_per_group);

    // A task's N side is its tile count, and the MMA runs it in nr_max-wide
    // panels, so N is billed at the next panel boundary: seven tiles a row (a
    // 28x28 plane under F(4,3)) costs what sixteen do, and the 1.78x the wider
    // tile saves on the multiply is exactly what the empty panel lanes give
    // back. A single tile row cannot fill a panel there, so the task spans
    // whole tile rows instead — the halo strip grows by one window per extra
    // row, and the row count is the one that fills the panels best rather than
    // the first that covers them: seven tiles want two rows (14 of 16) where
    // three would round to 21 of 32.
    // Only a row too narrow for one panel can want a second: once a row fills
    // it, widening to two rows pays nothing and multiplies the V, M and halo
    // footprints a task carries.
    const int64_t nr_max = (dt == DataType::f16) ? NR_MAX_F16 : NR_MAX_F32;
    const int64_t max_rows = std::min<int64_t>(TH, (nr_max + TW_pad - 1) / TW_pad);
    int64_t best_num = 0, best_den = 1;
    plan.tile_rows = 1;
    for (int64_t r = 1; r <= max_rows; ++r) {
        const int64_t n = r * TW_pad;
        const int64_t den = ((n + nr_max - 1) / nr_max) * nr_max;
        if (n * best_den > best_num * den) {
            best_num = n;
            best_den = den;
            plan.tile_rows = r;
        }
    }
    if (plan.tile_rows > 1) {
        plan.tile_block = plan.tile_rows * TW_pad;
    } else {
        plan.tile_block = std::clamp<int64_t>(TW_pad, 8, 16);
    }
    plan.num_tile_blocks = (plan.tile_rows > 1)
        ? 1
        : (TW_pad + plan.tile_block - 1) / plan.tile_block;

    // One task per (batch, group, tile-row group, tile block): those axes sit in
    // the grid rather than in a serial loop, so N·G also helps fill the pool.
    const int64_t NB = (in.dims[0] > 0) ? in.dims[0] : 1;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    const int64_t th_tasks = (TH + plan.tile_rows - 1) / plan.tile_rows;
    const int64_t plane_tasks = NB * G * th_tasks * plan.num_tile_blocks;

    // Second parallel axis: the filter rows, exactly as Conv2D's im2col plan
    // grows one (conv2d_implicit_gemm.cpp). A 14x14 output plane is 7 tile rows
    // and one 8-wide tile block, so the grid is 7 tasks and the tile axis has
    // nothing left to give, TW_pad already being one vector wide. Measured on
    // this host (bench_conv2d conv2d_wino_f32_14x14_256ch, ResNet-18's
    // layer3.x): 20 threads ran that layer slower than 4 — 0.59 vs 0.45 ms —
    // because 7 tasks over 20 workers leaves each alone on a core, the spread
    // over the slow E-cores decides the time, and im2col's 390 GFLOPS pulls
    // ahead of winograd's 198.
    //
    // A chunk costs the halo strip it rebuilds and the input transform it
    // repeats — the two things task-local scratch exists to do once per task.
    // That is IC·4·halo_ld copies plus ~32 adds per tile per channel against a
    // GEMM of 16·M·tiles·IC MACs: under a percent of the task, so it is worth
    // spending — but only down to the M floor below.
    //
    // The chunk count is the NEAREST whole number to nt/plane_tasks, not the
    // ceiling: past the pool a split buys nothing and costs a whole second wave.
    // 16 threads on this 7-task plane want 2 chunks (14 tasks, one wave) where
    // ceil picks 3 (21 tasks, one straggler) — 0.35 vs 0.48 ms on the case
    // above. Below 1.5·plane_tasks the round lands on 1, keeping the single
    // dispatch, which is what 8 threads want too. The floor is two M-panels.
    plan.oc_chunks = 1;
    if (plane_tasks < nt) {
        const int64_t mr = (dt == DataType::f16) ? MR_MAX_F16 : MR_MAX_F32;
        const int64_t max_chunks = std::max<int64_t>(1, oc_per_group / (2 * mr));
        const int64_t want = (nt + plane_tasks / 2) / plane_tasks;
        plan.oc_chunks = std::clamp<int64_t>(want, 1, max_chunks);
    }
    plan.num_tasks  = plane_tasks * plan.oc_chunks;
    // Padded columns per strip. A group transform reads one 16-wide vector past
    // its last tile — 2 columns on F(2,3) (a deinterleave), 4 on F(4,3) plus the
    // slack the double deinterleave needs. The tail group of a block smaller
    // than its own tile count reads past the strip into the next channel's, and
    // no write-back reaches those tiles (ow_end masks them).
    plan.halo_ld    = ktile * (plan.tile_rows > 1 ? TW_pad : plan.tile_block)
                    + (order == WinoOrder::F43 ? 8 : 2);
    // The window rows of a task: one tile row's window, plus a tile row per
    // extra row the task spans (each shares the two overlapping window rows).
    plan.halo_rows  = (plan.tile_rows - 1) * ktile + wino_alpha(order);

    plan.halo_size = ICG * plan.halo_rows * plan.halo_ld;
    // One extra vector of slack per transform position: the positions are
    // written one vector at a time, and a plane stride landing on a power-of-two
    // multiple collapses those stores onto a handful of L1 sets. Measured on
    // 1x64x56x56 f32 (interleaved A/B): 1.50 -> 1.32 ms at 1 thread, 0.50 ->
    // 0.47 at 4; f16 loses about a point at 1 thread, so the slack still pays.
    plan.v_p_stride = ICG * plan.tile_block + 8;
    plan.v_size     = wino_pos(order) * plan.v_p_stride;
    plan.m_size     = wino_pos(order) * plan.ocn_block * plan.tile_block;

    // Uniform packed-B scratch: one V slice [icn_block][tile_block] per slot.
    const int64_t esz    = static_cast<int64_t>(data_type_size(dt));
    const int64_t ldd_b_max = mma_pack_align(
        static_cast<size_t>(nr_max * plan.icn_block * esz)) / esz;
    plan.b_size = ((plan.tile_block + nr_max - 1) / nr_max) * ldd_b_max;

    plan.slot_size = plan.halo_size + plan.v_size + plan.m_size + plan.b_size;

    // Slots: one per worker when the caller reports worker ids, else one per
    // task — without a worker id no slot is safe to share. The count follows the
    // *workers*, not the reported thread count, so an over-subscribed grid (more
    // tasks than workers) does not multiply the arena.
    const int64_t avail = (num_slots > 0) ? num_slots : nt;
    plan.num_slots = (use_thread_slots && avail < plan.num_tasks) ? avail
                                                                  : plan.num_tasks;

    plan.workspace_size = static_cast<size_t>(plan.num_slots * plan.slot_size)
                        * data_type_size(dt);
    return plan;
}

}  // namespace channel_first

// =========================================================================
//  Channel-last route (channel-last)
// =========================================================================

WinoOrder wino_pick_order(DataType dt, int64_t ic_per_group, int64_t oc_per_group,
                          int64_t out_h, int64_t out_w, bool channel_last,
                          int num_threads) noexcept
{
    // f16 stays on F(2,3) on both routes: Bᵀ scales a window by up to 5 and Aᵀ
    // the products by 8, so the V and M rounds cost three bits on top of an f16
    // for an f16 GEMM accumulate. Measured 0.63-0.73x of F(2,3) on the
    // Channel-last route (0.56-0.87x at 1 thread), but at 0.125 of full scale on
    // a 32-channel case where the f16 contract holds 0.05 — the speed is there
    // and the accuracy is not.
    // A caller who wants it anyway pins winograd_order = 4.
    if (dt != DataType::f32) { return WinoOrder::F23; }

    // The U is streamed on every call, and F(4,3)'s is 2.25x the blob — 36
    // transform positions against 16 for the same filter. Past a 32 MB-class L3
    // (16 MB here, leaving room for the activations) that stream is the wall: a
    // 512x512 filter is 37.7 MB and measured 1.62x F(2,3) at 7x7 and 0.92x at
    // 28x28 at 1 thread on the channel-last route, 1.72x and 1.38x at 20 — the
    // one shape where the wider tile stops paying for itself. Below the fence it
    // pays.
    constexpr int64_t u_budget = 16 << 20;
    if (wino_pos(WinoOrder::F43) * ic_per_group * oc_per_group * 4 > u_budget) {
        return WinoOrder::F23;
    }

    // Channel-last reads its windows out of the packed panel in place, so the
    // dearer transform is the only cost the order adds: 0.61-0.92x of F(2,3) at
    // 1 thread over every plane from 28x28 up, 0.60-0.86x at 20 threads except
    // the two narrowest filters (32 and 64 channels on a 28x28 plane, 1.27x and
    // 1.28x, against 21 and 55 us of work). Below 28x28 the tile grid is a
    // handful of tasks — 16 tiles for a 14x14 plane — and the order loses both
    // ways (0.90-0.93x at 1 thread, 1.36-1.81x at 20).
    if (channel_last) {
        return (out_h >= 28 && out_w >= 28) ? WinoOrder::F43 : WinoOrder::F23;
    }

    // Channel-first pays a halo strip per task instead — ICG · window rows · window
    // columns, six wide and twelve deep for a task of two tile rows against
    // F(2,3)'s two and six — and 4x4 tiles quarter the tile grid the strip is
    // rebuilt for. On a small team the tile axes fill the pool by themselves, so
    // the strip is amortized all the way down to 28x28: measured 0.71-0.81x of
    // F(2,3) at 1 thread on 256ch and 0.73-0.75x at 4, 0.71-0.80x on 56x56.
    // Past four workers the grid is short — 14 tasks on a 56x56 plane, 7 on a
    // 28x28 — and the oc-chunk axis that makes up the difference rebuilds that
    // strip per chunk: 28x28 drifts to 1.04x at six threads and 5.1x at 20, so
    // the floor climbs with the team rather than sitting at the widest case.
    if (ic_per_group < 64) { return WinoOrder::F23; }
    const int64_t floor = num_threads <= 4 ? 28 : num_threads <= 8 ? 56 : 112;
    if (out_h < floor || out_w < floor) { return WinoOrder::F23; }
    return WinoOrder::F43;
}

WinoChannelLastU wino_channel_last_u(int64_t ic_per_group, int64_t oc_per_group,
                                     int64_t groups, DataType dt,
                                     WinoOrder order) noexcept {
    WinoChannelLastU u;
    if (ic_per_group <= 0 || oc_per_group <= 0 || groups <= 0) { return u; }
    if (dt != DataType::f32 && dt != DataType::f16) { return u; }
    u.order = order;
    u.pos   = wino_pos(order);

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
    u.total = u.pos * u.plane;
    return u;
}

namespace channel_last {
namespace {

using namespace nnops::simd;

/// Tiles per task. A slot holds the task's whole V and M — pos·T·(IC/G + ocn)
/// elements, 150-500 KiB (f32) over the 32-256 channel shapes Winograd is used
/// on — so this is the knob that decides whether they stay in L2. 24 is a
/// multiple of mr_max on both arches (6 on x86, 8 on aarch64), so a full task
/// leaves no partial A panel.
///
/// F(4,3) holds 2.25x the V per tile and produces 4x the outputs, so the same
/// scratch budget is about half as many tiles. Task count, not scratch, is what
/// sets it: 12 tiles is 192 outputs, still a whole tile row of a 14x14 plane.
constexpr int64_t kTileBlock(WinoOrder o) noexcept {
    return o == WinoOrder::F43 ? 12 : 24;
}
/// Growth stops here: below it a task re-reads its whole U slice (pos·kc·oc)
/// for too few tiles of MACs, and the per-task transform cost shows.
constexpr int64_t kTileBlockMin(WinoOrder o) noexcept {
    return o == WinoOrder::F43 ? 6 : 8;
}
/// Output channels per GEMM (the N side), snapped down to whole panels — the
/// same trade as the channel-first M block: each extra oc block is another pass over
/// the task's tiles (accumulator traffic plus one more B read).
constexpr int64_t kOcnBlock = 64;

/// Channel-last kernel: per-slot scratch is V then M, and U is read in place.
/// The transform order is a template parameter because it fixes the position
/// count, the tile size and the two transforms' shapes — everything else about
/// the task is order-independent.
template <class T, WinoOrder O>
void conv2d_winograd_channel_last_impl(const Conv2DAttributes& attrs,
                               TensorView& output,
                               std::span<const TensorView> inputs,
                               const ComputeContext& ctx,
                               const Conv2DWinogradPlan& plan,
                               void* workspace)
{
    constexpr int64_t kpos  = wino_pos(O);
    constexpr int64_t ktile = wino_tile(O);
    const auto& input = inputs[0];
    const auto& u_tensor = inputs[1];  // prepacked transform-domain weight
    const bool has_bias = inputs.size() > 2;

    // Logical [N, C, H, W]; channel-last stores [N][H][W][C], so a "row" is one spatial
    // line and a pixel's channels are contiguous.
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

    // U: [pos][nb][G·jb][ldd] — the GEMM's B panels, one set per transform
    // position, and the dims must match the geometry prepack derived them under.
    const WinoChannelLastU& ug = plan.u;
    NNOPS_ASSERT(u_tensor.shape(0) == kpos && u_tensor.shape(1) == ug.nb &&
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

    // One channel block is one vector — 8 lanes for both dtypes (NEON f32 is
    // two float32x4_t behind the same abstraction).
    constexpr int64_t vec = simd_lane_for<T>;

    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && plan.num_slots < plan.num_tasks;

    // One task = one (batch, group, tile block, oc chunk): transform its tiles
    // into V once, run the pos GEMMs for its oc blocks into M, write them back.
    const auto run_task = [&](int64_t t) {
        int64_t idx = t;
        // The oc chunk varies fastest, as in the channel-first route: the chunks of one
        // plane are adjacent in task order and re-read the same input lines, so
        // whichever arrives second finds them in L2.
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

        // Input transform, tile by tile: the tile's window is swept across the
        // channel blocks while its input lines are still hot.
        for (int64_t tl = 0; tl < ntl; ++tl) {
            const int64_t tt = tile0 + tl;
            const int64_t th = tt / TW;
            const int64_t tw = tt - th * TW;
            const int64_t ih0 = ktile * th - PH;
            const int64_t iw0 = ktile * tw - PW;
            for (int64_t ic = 0; ic < ic_per_group; ic += vec) {
                const int lanes = static_cast<int>(std::min<int64_t>(vec, ic_per_group - ic));
                // The block's channel offset goes into both pointers: V's row
                // channel and the activation's channel lane.
                T* v_tile = v_all + tl * v_lda + ic;
                if constexpr (O == WinoOrder::F23) {
                    k::winograd_input_channel_last<T>(task_in + ic, in_row, IC,
                                              static_cast<int>(ih0), static_cast<int>(iw0),
                                              static_cast<int>(IH), static_cast<int>(IW),
                                              v_tile, v_p_stride, lanes);
                } else {
                    k::winograd43_input_channel_last<T>(task_in + ic, in_row, IC,
                                                static_cast<int>(ih0), static_cast<int>(iw0),
                                                static_cast<int>(IH), static_cast<int>(IW),
                                                v_tile, v_p_stride, lanes);
                }
            }
        }

        // This chunk's oc blocks, disjoint; splitting on block boundaries rather
        // than column ranges keeps every start a whole panel into the prepacked U.
        const int64_t b0 = chunk * oc_blocks / oc_chunks;
        const int64_t b1 = (chunk + 1) * oc_blocks / oc_chunks;

        for (int64_t b = b0; b < b1; ++b) {
            const int64_t ocn0 = b * ocn_block;
            const int64_t ocnc = std::min(ocn_block, oc_per_group - ocn0);

            // M_p zeroes on the first k-block and accumulates after; bias is
            // added at write-back, never here (Aᵀ(·)A would distort it).
            for (int64_t kb = 0; kb < ug.nb; ++kb) {
                const int64_t kc = ug.kc;
                const T* b_base = u_ptr + (kb * ug.jbg + g * ug.jb + ocn0 / ug.nr) * ug.ldd;
                const T* a_base = v_all + kb * kc;
                for (int p = 0; p < kpos; ++p) {
                    tile_mma_direct(static_cast<int>(ntl), static_cast<int>(ocnc),
                                    static_cast<int>(kc),
                                    m_all + p * m_p_stride,
                                    static_cast<int>(m_oc_stride),
                                    a_base + p * v_p_stride, static_cast<int>(v_lda),
                                    b_base + p * ug.plane, /*ldb=*/-1, /*stage=*/nullptr,
                                    -std::numeric_limits<float>::infinity(),
                                    std::numeric_limits<float>::infinity(),
                                    /*zero_mode=*/kb == 0);
                }
            }

            // Output transform + bias + epilogue + write-back, per tile: its
            // tile×tile pixels are adjacent in memory, and a short output extent
            // leaves the pixels past it with a null destination.
            for (int64_t tl = 0; tl < ntl; ++tl) {
                const int64_t tt = tile0 + tl;
                const int64_t th = tt / TW;
                const int64_t tw = tt - th * TW;
                const int64_t oh = ktile * th;
                const int64_t ow = ktile * tw;
                for (int64_t oc = 0; oc < ocnc; oc += vec) {
                    const int lanes = static_cast<int>(std::min<int64_t>(vec, ocnc - oc));
                    T* base = task_out + oh * out_row + ow * OC + ocn0 + oc;
                    T* dst[ktile][ktile];
                    for (int r = 0; r < ktile; ++r) {
                        for (int c = 0; c < ktile; ++c) {
                            dst[r][c] = (oh + r < OH && ow + c < OW)
                                      ? base + r * out_row + c * OC : nullptr;
                        }
                    }
                    const T* bias_oc = b_ptr ? b_ptr + g * oc_per_group + ocn0 + oc
                                             : nullptr;
                    const T* m_tile = m_all + tl * m_oc_stride + oc;
                    if constexpr (O == WinoOrder::F23) {
                        k::winograd_output_channel_last<T>(m_tile, m_p_stride, dst,
                                                   attrs.epilogue, bias_oc,
                                                   attrs.add_to, lanes);
                    } else {
                        k::winograd43_output_channel_last<T>(m_tile, m_p_stride, dst,
                                                     attrs.epilogue, bias_oc,
                                                     attrs.add_to, lanes);
                    }
                }
            }
        }
    };

    ctx.cpu.run(0, plan.num_tasks, run_task);
}

/// Resolve the channel-last route: the activation is channel-last and U is prepacked as
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

    // k3s1 only — the F(2,3) derivation assumes a 3×3, unit-stride, undilated
    // kernel.
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

    // The transform order is the blob's leading dimension: 16 positions is
    // F(2,3), 36 is F(4,3). Nothing else can tell them apart, and reading it off
    // the U is what lets the same Conv2D compute both.
    const WinoOrder order = (u.dims[0] == wino_pos(WinoOrder::F43)) ? WinoOrder::F43
                                                                   : WinoOrder::F23;
    // The panels' geometry is derived, not read: the same function ran at prepack
    // time, so matching it against the descriptor keeps the two ends from drifting.
    const WinoChannelLastU ug = wino_channel_last_u(ic_per_group, oc_per_group, G, dt,
                                                    order);
    if (ug.total == 0 || u.dims[0] != ug.pos || u.dims[1] != ug.nb ||
        u.dims[2] != ug.jbg || u.dims[3] != ug.ldd) { return plan; }
    plan.u = ug;
    plan.order = order;

    const int64_t OH = out.dims[2];
    const int64_t OW = out.dims[3];
    if (OH <= 0 || OW <= 0) { return plan; }

    const int64_t tile = wino_tile(order);
    const int64_t TH = (OH + tile - 1) / tile;
    const int64_t TW = (OW + tile - 1) / tile;
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

    // Tile blocks first — flattening repeats no work — then the oc axis, which
    // repeats the input transform (its input lines stay hot) but nothing of U.
    const int64_t NB = (in.dims[0] > 0) ? in.dims[0] : 1;
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    const int64_t tb_target = kTileBlock(order);
    int64_t ntb = (tiles + tb_target - 1) / tb_target;
    while (NB * G * ntb < nt && tiles / (ntb + 1) >= kTileBlockMin(order)) { ++ntb; }
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

    // V: one [tile][ic] plane per position, tiles contiguous inside a position
    // (that is the GEMM's A, so lda is a tile row). The row is padded to the
    // vector width so a tail block can be written whole, and the +8 per position
    // keeps the 16 planes off one L1 set — the channel-first route's measurement holds.
    plan.v_lda      = ((ic_per_group + 7) / 8) * 8;
    plan.v_p_stride = plan.tile_block * plan.v_lda + 8;
    plan.v_size     = ug.pos * plan.v_p_stride;
    // M: [tile][oc] per position, a whole panel per row so the MMA's widest row
    // read stays inside it.
    plan.m_oc_stride = plan.ocn_block;
    plan.m_size      = ug.pos * plan.tile_block * plan.m_oc_stride;

    const int64_t esz = static_cast<int64_t>(data_type_size(dt));
    plan.halo_size = 0;   // no halo strip on this route
    plan.b_size    = 0;   // and no packed-B block: U is read in place
    // Round the slot to 16 elements so every slot base keeps the transform's
    // 32-byte f16 stores aligned.
    plan.slot_size = ((plan.v_size + plan.m_size) + 15) / 16 * 16;
    // The slot count follows the *workers*, not the reported thread count: an
    // over-subscribed grid would otherwise multiply the arena for nothing.
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
    // interchangeable, so a mismatched pair must resolve to a zeroed plan (and
    // the kernel's assert) rather than to a route that would index it wrongly.
    if (!inputs.empty() && is_channel_last(inputs[0].layout)) {
        return channel_last::make_plan(attrs, inputs, outputs, num_threads,
                                       use_thread_slots, num_slots);
    }
    return channel_first::make_plan(attrs, inputs, outputs, num_threads, use_thread_slots,
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
    // Never fall back to the reference conv2d here: U is not an OIHW filter —
    // Channel-first's is rank-3 with dims[0] == 16, which the reference reads as an OC
    // of 16.
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
    // Slot sizing: per worker when the caller reports worker ids; a single slot
    // when there is no pool (a serial run reuses it task after task); else one per
    // task (see Conv2DWinogradPlan::num_slots).
    const bool serial = !ctx.cpu.parallel_for;
    const Conv2DWinogradPlan plan =
        get_conv2d_winograd_plan(attrs, descs, outs, ctx.cpu.thread_count(),
                                 serial || ctx.cpu.thread_slot_count() > 0,
                                 ctx.cpu.thread_slot_count());
    NNOPS_ASSERT(plan.workspace_size > 0 &&
                 "conv2d_winograd_kernel: unsupported conv geometry (k3s1 only)");

    PoolPtr scratch(plan.workspace_size);
    const bool channel_last = plan.route == Conv2DWinogradRoute::ChannelLast;
    const bool f43  = plan.order == WinoOrder::F43;
    if (dt == DataType::f32) {
        if (channel_last) {
            if (f43) {
                channel_last::conv2d_winograd_channel_last_impl<float, WinoOrder::F43>(
                    attrs, output, inputs, ctx, plan, scratch.as<float>());
            } else {
                channel_last::conv2d_winograd_channel_last_impl<float, WinoOrder::F23>(
                    attrs, output, inputs, ctx, plan, scratch.as<float>());
            }
        } else if (f43) {
            channel_first::conv2d_winograd_impl<float, WinoOrder::F43>(
                attrs, output, inputs, ctx, plan, scratch.as<float>());
        } else {
            channel_first::conv2d_winograd_impl<float, WinoOrder::F23>(
                attrs, output, inputs, ctx, plan, scratch.as<float>());
        }
    } else if (channel_last) {
        if (f43) {
            channel_last::conv2d_winograd_channel_last_impl<half, WinoOrder::F43>(
                attrs, output, inputs, ctx, plan, scratch.as<half>());
        } else {
            channel_last::conv2d_winograd_channel_last_impl<half, WinoOrder::F23>(
                attrs, output, inputs, ctx, plan, scratch.as<half>());
        }
    } else if (f43) {
        channel_first::conv2d_winograd_impl<half, WinoOrder::F43>(
            attrs, output, inputs, ctx, plan, scratch.as<half>());
    } else {
        channel_first::conv2d_winograd_impl<half, WinoOrder::F23>(
            attrs, output, inputs, ctx, plan, scratch.as<half>());
    }
    return Status::ok;
}

}  // namespace nnops::backend::cpu
