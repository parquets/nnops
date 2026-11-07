/// @file recurrent_step.cpp
/// @brief One phase of one timestep on the fp MMA, and the chunked input
///        projection that feeds it.
///
/// The recurrent product H_{t-1}·R^T runs B rows at a time into the chunk
/// buffer, and the tile epilogue fuses everything after it: activation, cell
/// update, sequence-length mask and the Y store, so a gate value never leaves
/// cache. The projection is the same tile kernel with `rows·B` rows in m.
///
/// Layout of this file: the epilogue is instantiated per (dtype, phase) and is
/// also called by the int8 twin in recurrent_q8.cpp, so it has external
/// linkage; the per-phase step loop and its float/half switch stay local.

#include "backend/cpu/rnn/recurrent_internal.h"

namespace nnops::backend::cpu {

namespace {

// =========================================================================
//  Activations (the ONNX `activations` list) — per-element, so one vector
//  kernel serves every gate of every phase.
// =========================================================================

/// The pre-activation bound ONNX calls `clip`; 0 disables it.
inline v_f32x8 clipv(const v_f32x8& x, float clip) noexcept {
    if (clip <= 0.0f) { return x; }
    return v_min(v_max(x, v_set1_f32x8(-clip)), v_set1_f32x8(clip));
}

/// Only the five activations with a kernel here are accepted. ScaledTanh and
/// HardSigmoid — the two that consume `beta` — are rejected at the operator.
inline v_f32x8 actv(const v_f32x8& x, RnnActivation a, float alpha) noexcept {
    switch (a) {
    case RnnActivation::Sigmoid:   return kernel::v_sigmoid(x, v_set1_f32x8(1.0f));
    case RnnActivation::Tanh:      return kernel::v_tanh(x);
    case RnnActivation::Relu:      return v_max(x, v_zero_f32x8());
    case RnnActivation::LeakyRelu:
        return kernel::v_leaky_relu(x, v_zero_f32x8(), v_set1_f32x8(alpha));
    case RnnActivation::Elu:
        return kernel::v_elu(x, v_zero_f32x8(), v_set1_f32x8(alpha),
                             v_set1_f32x8(1.0f));
    }
    return x;
}

/// Store eight lanes into a Y row, which is `hidden` wide rather than hp.
template <typename T>
inline void store_y(T* y_row, int64_t h0, int64_t hidden, const v_f32x8& v) {
    if (h0 + 8 <= hidden) { vstore8(y_row + h0, v); return; }
    alignas(32) T tmp[8];
    vstore8(tmp, v);
    for (int64_t j = 0; j < hidden - h0; ++j) { y_row[h0 + j] = tmp[j]; }
}

/// One vector of the gate row with the phase bias folded in. The bias is added
/// here rather than seeded into the projection so the epilogue is the only pass
/// that touches the gate row twice, and the bias vector stays in L1.
template <typename T>
inline v_f32x8 load_g(const T* row, const T* bias, int64_t idx)
{
    const v_f32x8 v = vload8(row + idx);
    return bias ? v_add(v, vload8(bias + idx)) : v;
}

/// The h gate's pre-activation: the projection, plus r ⊙ (Ht-1·Rh^T + Rbh) when
/// linear_before_reset kept the recurrent product separate.
template <typename T>
inline v_f32x8 gate_pre(const v_f32x8& vg, const T* rec, const T* rh_bias,
                        const v_f32x8& vr, int64_t cg)
{
    if (rec == nullptr) { return vg; }
    v_f32x8 vrr = vload8(rec + cg);
    if (rh_bias != nullptr) { vrr = v_add(vrr, vload8(rh_bias + cg)); }
    return v_fmadd(vr, vrr, vg);
}

}  // namespace

/// The fused epilogue over one tile: gate math, state update, Y store.
template <typename T, PhaseKind Kind>
void epilogue_tile(const StepCtx<T>& s, int64_t m0, int64_t mcnt,
                   int64_t n0, int64_t ncnt)
{
    constexpr int64_t Q = 8 * phase_gates(Kind);
    const v_f32x8 one = v_set1_f32x8(1.0f);
    const T* g_tile = s.in_proj + m0 * s.ldc + n0;
    const T* r_tile = s.rec ? s.rec + m0 * s.ldc + n0 : nullptr;

    for (int64_t m = 0; m < mcnt; ++m) {
        const int64_t b = m0 + m;
        const bool active = (s.seq_len == nullptr) || (s.t < s.seq_len[b]);

        if (!active) {
            const T* hp_row = s.h_prev + b * s.hp;
            T* hn_row = s.h_next + b * s.hp;
            const T* cp_row = s.c_prev ? s.c_prev + b * s.hp : nullptr;
            T* cn_row = s.c_next ? s.c_next + b * s.hp : nullptr;
            T* y_row = s.y_row0 ? s.y_row0 + b * s.y_stride : nullptr;
            for (int64_t cg = 0; cg < ncnt; cg += Q) {
                const int64_t h0 = (n0 + cg) / Q * 8;
                for (int64_t j = 0; j < 8; ++j) {
                    hn_row[h0 + j] = hp_row[h0 + j];
                    if (cn_row) { cn_row[h0 + j] = cp_row[h0 + j]; }
                }
                if (y_row) {
                    const int64_t hn = std::min<int64_t>(8, s.hidden - h0);
                    for (int64_t j = 0; j < hn; ++j) { s_store(y_row + h0 + j, 0.0f); }
                }
            }
            continue;
        }

        const T* g_row = g_tile + m * s.ldc;
        const T* r_row = r_tile ? r_tile + m * s.ldc : nullptr;
        const T* hp_row = s.h_prev + b * s.hp;
        T* hn_row = s.h_next + b * s.hp;
        const T* cp_row = s.c_prev ? s.c_prev + b * s.hp : nullptr;
        T* cn_row = s.c_next ? s.c_next + b * s.hp : nullptr;
        T* y_row = s.y_row0 ? s.y_row0 + b * s.y_stride : nullptr;

        for (int64_t cg = 0; cg < ncnt; cg += Q) {
            const int64_t h0 = (n0 + cg) / Q * 8;

            if constexpr (Kind == PhaseKind::Rnn) {
                const v_f32x8 v = actv(clipv(load_g(g_row, s.bias, cg), s.clip),
                                       s.act_g, s.alpha_g);
                vstore8(hn_row + h0, v);
                if (y_row) { store_y(y_row, h0, s.hidden, v); }

            } else if constexpr (Kind == PhaseKind::Lstm) {
                // ONNX gate order is i, o, f, c.
                const v_f32x8 vi = actv(clipv(load_g(g_row, s.bias, cg + 0 * 8), s.clip),
                                        s.act_f, s.alpha_f);
                const v_f32x8 vo = actv(clipv(load_g(g_row, s.bias, cg + 1 * 8), s.clip),
                                        s.act_f, s.alpha_f);
                const v_f32x8 vf =
                    s.input_forget
                        ? v_sub(one, vi)
                        : actv(clipv(load_g(g_row, s.bias, cg + 2 * 8), s.clip),
                               s.act_f, s.alpha_f);
                const v_f32x8 vc = actv(clipv(load_g(g_row, s.bias, cg + 3 * 8), s.clip),
                                        s.act_g, s.alpha_g);
                const v_f32x8 cell = v_fmadd(vf, vload8(cp_row + h0), v_mul(vi, vc));
                vstore8(cn_row + h0, cell);
                const v_f32x8 v = v_mul(vo, actv(cell, s.act_h, s.alpha_h));
                vstore8(hn_row + h0, v);
                if (y_row) { store_y(y_row, h0, s.hidden, v); }

            } else if constexpr (Kind == PhaseKind::GruAll) {
                // ONNX gate order is z, r, h; h~'s recurrent term is the one r
                // scales, so it waits for r here.
                v_f32x8 vz = v_add(load_g(g_row, s.bias, cg + 0 * 8),
                                   vload8(r_row + cg + 0 * 8));
                v_f32x8 vr = v_add(load_g(g_row, s.bias, cg + 1 * 8),
                                   vload8(r_row + cg + 1 * 8));
                v_f32x8 vh = load_g(g_row, s.bias, cg + 2 * 8);

                vz = actv(clipv(vz, s.clip), s.act_f, s.alpha_f);
                vr = actv(clipv(vr, s.clip), s.act_f, s.alpha_f);
                vh = gate_pre(vh, r_row, s.rh_bias, vr, cg + 2 * 8);
                vh = actv(clipv(vh, s.clip), s.act_g, s.alpha_g);
                const v_f32x8 out = v_fmadd(vz, vload8(hp_row + h0),
                                            v_mul(v_sub(one, vz), vh));
                vstore8(hn_row + h0, out);
                if (y_row) { store_y(y_row, h0, s.hidden, out); }

            } else if constexpr (Kind == PhaseKind::GruZr) {
                // z, and r already blended with Ht-1: the h~ phase's operand.
                const v_f32x8 vz = actv(clipv(load_g(g_row, s.bias, cg + 0 * 8), s.clip),
                                        s.act_f, s.alpha_f);
                const v_f32x8 vr = actv(clipv(load_g(g_row, s.bias, cg + 1 * 8), s.clip),
                                        s.act_f, s.alpha_f);
                vstore8(s.zb + b * s.hp + h0, vz);
                vstore8(s.aux + b * s.hp + h0, v_mul(vr, vload8(hp_row + h0)));

            } else {  // GruH
                const v_f32x8 vh = actv(clipv(load_g(g_row, s.bias, cg), s.clip),
                                        s.act_g, s.alpha_g);
                const v_f32x8 vz = vload8(s.zb + b * s.hp + h0);
                const v_f32x8 out = v_fmadd(vz, vload8(hp_row + h0),
                                            v_mul(v_sub(one, vz), vh));
                vstore8(hn_row + h0, out);
                if (y_row) { store_y(y_row, h0, s.hidden, out); }
            }
        }
    }
}

template void epilogue_tile<float, PhaseKind::Rnn>(
    const StepCtx<float>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<float, PhaseKind::Lstm>(
    const StepCtx<float>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<float, PhaseKind::GruAll>(
    const StepCtx<float>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<float, PhaseKind::GruZr>(
    const StepCtx<float>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<float, PhaseKind::GruH>(
    const StepCtx<float>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<half, PhaseKind::Rnn>(
    const StepCtx<half>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<half, PhaseKind::Lstm>(
    const StepCtx<half>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<half, PhaseKind::GruAll>(
    const StepCtx<half>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<half, PhaseKind::GruZr>(
    const StepCtx<half>&, int64_t, int64_t, int64_t, int64_t);
template void epilogue_tile<half, PhaseKind::GruH>(
    const StepCtx<half>&, int64_t, int64_t, int64_t, int64_t);

namespace {

/// The recurrent product for one timestep, then the epilogue, over the tile grid.
/// With `rec` set the tile is built from nothing (linear_before_reset's second
/// accumulation); otherwise it adds into the projection.
template <typename T, PhaseKind Kind>
void run_phase_kernel(const StepCtx<T>& s, const ComputeContext& ctx)
{
    const int64_t tasks = s.m_blocks * s.n_blocks;
    const bool fresh = (s.rec != nullptr);

    auto body = [&](int64_t task) {
        const int64_t mblk = task / s.n_blocks;
        const int64_t nblk = task % s.n_blocks;
        const int64_t m0 = mblk * s.mc;
        const int64_t mcnt = std::min<int64_t>(s.mc, s.batch - m0);
        const int64_t n0 = nblk * s.nc;
        const int64_t ncnt = std::min<int64_t>(s.nc, s.cols - n0);
        if (mcnt <= 0 || ncnt <= 0) { return; }

        T* c_tile = (fresh ? s.rec : s.in_proj) + m0 * s.ldc + n0;
        const T* a = s.a + m0 * s.hp;
        const T* b_panel = s.packed_r + (n0 / s.r_nr) * s.r_ldd;

        for (int64_t k = 0; k < s.hidden; k += s.kc) {
            const int64_t kc = std::min<int64_t>(s.kc, s.hidden - k);
            tile_mma_direct(static_cast<int>(mcnt), static_cast<int>(ncnt),
                                static_cast<int>(kc),
                                c_tile, static_cast<int>(s.ldc),
                                a + k, static_cast<int>(s.hp),
                                b_panel + k * s.r_nr, -s.r_ldd, /*stage=*/nullptr,
                                -INFINITY, INFINITY, k == 0 && fresh);
        }

        // The tile is offset to n0 but the bias vectors are not, and the epilogue
        // indexes both by the same block-local column.
        StepCtx<T> tile = s;
        tile.bias = s.bias ? s.bias + n0 : nullptr;
        tile.rh_bias = s.rh_bias ? s.rh_bias + n0 : nullptr;
        epilogue_tile<T, Kind>(tile, m0, mcnt, n0, ncnt);
    };

    if (tasks <= 1) {
        body(0);
    } else {
        ctx.cpu.run(0, tasks, body);
    }
}

}  // namespace

void run_phase(const StepCtx<float>& s, PhaseKind kind, const ComputeContext& ctx) {
    switch (kind) {
    case PhaseKind::Rnn:    run_phase_kernel<float, PhaseKind::Rnn>(s, ctx); break;
    case PhaseKind::Lstm:   run_phase_kernel<float, PhaseKind::Lstm>(s, ctx); break;
    case PhaseKind::GruAll: run_phase_kernel<float, PhaseKind::GruAll>(s, ctx); break;
    case PhaseKind::GruZr:  run_phase_kernel<float, PhaseKind::GruZr>(s, ctx); break;
    case PhaseKind::GruH:   run_phase_kernel<float, PhaseKind::GruH>(s, ctx); break;
    }
}

void run_phase(const StepCtx<half>& s, PhaseKind kind, const ComputeContext& ctx) {
    switch (kind) {
    case PhaseKind::Rnn:    run_phase_kernel<half, PhaseKind::Rnn>(s, ctx); break;
    case PhaseKind::Lstm:   run_phase_kernel<half, PhaseKind::Lstm>(s, ctx); break;
    case PhaseKind::GruAll: run_phase_kernel<half, PhaseKind::GruAll>(s, ctx); break;
    case PhaseKind::GruZr:  run_phase_kernel<half, PhaseKind::GruZr>(s, ctx); break;
    case PhaseKind::GruH:   run_phase_kernel<half, PhaseKind::GruH>(s, ctx); break;
    }
}

/// The input projection over one chunk — the same tile kernel the recurrent
/// product uses, only with `rows·B` rows in m instead of B. That is what makes a
/// recurrent layer's projection cost a GEMM's rather than a GEMV's: every
/// timestep of the chunk is already known, so its rows fill the tile.
///
/// The bias is not seeded here; the epilogue adds it.
template <typename T>
void project_chunk(const T* x, int64_t x_ld, T* c, int64_t ldc, const T* packed_w,
                   int64_t w_ldd, int64_t w_nr, int64_t rows, int64_t cols,
                   int64_t input, int64_t kc, int64_t nc, const ComputeContext& ctx)
{
    // Wide row blocks, since a chunk's m is hundreds of rows; then split them
    // until the grid is deep enough to keep every worker busy.
    const int64_t threads = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    const int64_t n_blocks = (cols + nc - 1) / nc;
    int64_t mc = 64;
    for (;;) {
        const int64_t mb = (rows + mc - 1) / mc;
        if (mb * n_blocks >= threads || mc <= 8) { break; }
        mc -= 8;
    }
    const int64_t m_blocks = (rows + mc - 1) / mc;

    auto body = [&](int64_t task) {
        const int64_t m0 = (task / n_blocks) * mc;
        const int64_t n0 = (task % n_blocks) * nc;
        const int64_t mcnt = std::min<int64_t>(mc, rows - m0);
        const int64_t ncnt = std::min<int64_t>(nc, cols - n0);
        if (mcnt <= 0 || ncnt <= 0) { return; }

        T* c_tile = c + m0 * ldc + n0;
        const T* a = x + m0 * x_ld;
        const T* b_panel = packed_w + (n0 / w_nr) * w_ldd;

        for (int64_t k = 0; k < input; k += kc) {
            const int64_t kcnt = std::min<int64_t>(kc, input - k);
            tile_mma_direct(static_cast<int>(mcnt), static_cast<int>(ncnt),
                                static_cast<int>(kcnt),
                                c_tile, static_cast<int>(ldc),
                                a + k, static_cast<int>(x_ld),
                                b_panel + k * w_nr, -w_ldd, /*stage=*/nullptr,
                                -INFINITY, INFINITY, k == 0);
        }
    };

    const int64_t tasks = m_blocks * n_blocks;
    if (tasks <= 1) {
        body(0);
    } else {
        ctx.cpu.run(0, tasks, body);
    }
}

template void project_chunk<float>(const float*, int64_t, float*, int64_t,
                                   const float*, int64_t, int64_t, int64_t,
                                   int64_t, int64_t, int64_t, int64_t,
                                   const ComputeContext&);
template void project_chunk<half>(const half*, int64_t, half*, int64_t,
                                  const half*, int64_t, int64_t, int64_t,
                                  int64_t, int64_t, int64_t, int64_t,
                                  const ComputeContext&);

}  // namespace nnops::backend::cpu
