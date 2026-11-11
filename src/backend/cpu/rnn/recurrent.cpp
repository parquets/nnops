/// @file recurrent.cpp
/// @brief The CPU recurrent driver — RNN, GRU and LSTM behind one timeline.
///
/// A recurrent layer's time splits three ways, and only the middle one is a GEMM
/// that has to run a step at a time:
///
///   1. The input projection X·W^T. All of its rows exist up front, so it runs
///      as one GEMM per chunk of timesteps (M = chunk·B) and pays the GEMM's
///      arithmetic intensity rather than a vector-matrix product's.
///      (recurrent_step.cpp / recurrent_q8.cpp)
///   2. H_{t-1}·R^T. Serial in t by definition, so this one does run B rows at a
///      time — but into the chunk buffer with the prepacked R panels, and it is
///      the only per-step pass over [B, gates·H].
///   3. The gate arithmetic. O(gates·H) flops per row: trivial, and ruinous as a
///      separate pass. It is fused into the tile epilogue, so a gate value goes
///      from the MMA accumulator to the activation, the cell update, the
///      sequence-length mask and the Y store without leaving cache.
///
/// Gate layout. One block of eight hidden units of a phase is interleaved as
/// [g0·h0..h7][g1·h0..h7]...[gG-1·h0..h7], so every gate of eight hidden units
/// is one contiguous SIMD vector and the epilogue never shuffles lanes. A phase
/// is 8·gates·⌈H/8⌉ columns wide, and column tiles snap to lcm(8·gates, nr_max),
/// which lands each tile on both a hidden-block boundary (so the epilogue is
/// uniform) and a packed-RHS panel boundary (so the MMA can address it as a
/// panel index).
///
/// The hidden state is padded to whole hidden blocks, so nothing in the step
/// loop has a tail except the Y store, which is the caller's width.
///
/// A GRU is one pass per step only under linear_before_reset = 1, where h~'s
/// recurrent term is r ⊙ (H_{t-1}·Rh^T + Rbh) and so takes H_{t-1} like the other
/// two gates: one tile carries z, r and h~, the recurrent product accumulates
/// into a second buffer, and r combines them in the epilogue. Without it that
/// term is (r ⊙ H_{t-1})·Rh^T, whose operand does not exist until r does, so the
/// step takes two passes — z and r, then h~ — and the first leaves both z and
/// r ⊙ H_{t-1} behind for the second.
///
/// sequence_lens: a row past its length holds its state and writes Y = 0, and a
/// length-0 sequence gets a zeroed final state — ONNX's null_state_default_value.
///
/// The file split: recurrent_internal.h holds what the pieces agree on,
/// recurrent_plan.cpp the tile/chunk decision, recurrent_pack.cpp the blob,
/// recurrent_step.cpp the fp kernels and recurrent_q8.cpp the int8 twin. This
/// file walks the timeline, in four pieces: prepare_run resolves the plan, the
/// operand views and the blob header; scratch_size counts the pooled block;
/// run_direction packs one direction's weights and walks its chunks; run_step
/// marshals one timestep's StepCtx. recurrent_impl is then the loop over
/// directions.

#include "backend/cpu/rnn/recurrent_internal.h"

namespace nnops::backend::cpu {

namespace {

// =========================================================================
//  Scratch carving
// =========================================================================

/// A bump allocator over one pooled block. Every region starts 64-byte aligned,
/// which is both the pool's granularity and the panels' (mma_pack_align).
class Arena {
public:
    Arena(void* base, size_t bytes)
        : cur_(static_cast<char*>(base)), end_(cur_ + bytes) {}

    template <typename T>
    T* take(int64_t count) {
        cur_ = reinterpret_cast<char*>(
            align_up<64>(reinterpret_cast<uintptr_t>(cur_)));
        char* const p = cur_;
        cur_ += static_cast<size_t>(count) * sizeof(T);
        NNOPS_ASSERT(cur_ <= end_ && "recurrent scratch exhausted");
        return reinterpret_cast<T*>(p);
    }

    /// For the quantized route's byte-sized regions — the s8 panels are one
    /// byte per element, and their stride is a byte count.
    char* take_bytes(int64_t bytes) {
        cur_ = reinterpret_cast<char*>(
            align_up<64>(reinterpret_cast<uintptr_t>(cur_)));
        char* const p = cur_;
        cur_ += static_cast<size_t>(std::max<int64_t>(bytes, 0));
        NNOPS_ASSERT(cur_ <= end_ && "recurrent scratch exhausted");
        return p;
    }

private:
    char* cur_;
    char* end_;
};

/// A [B, H] slice of a [.., B, H] tensor at direction `d`, widened into a
/// [B, hp] buffer with the padding lanes zeroed. A null source is all zeros.
template <typename T>
void load_state(const TensorView* src, int64_t d, int64_t batch, int64_t hidden,
                int64_t hp, T* dst)
{
    for (int64_t b = 0; b < batch; ++b) {
        T* row = dst + b * hp;
        for (int64_t h = hidden; h < hp; ++h) { s_store(row + h, 0.0f); }
        if (src) {
            std::memcpy(row, src->ptr<T>(static_cast<size_t>(d * batch + b)),
                        static_cast<size_t>(hidden) * sizeof(T));
        } else {
            for (int64_t h = 0; h < hidden; ++h) { s_store(row + h, 0.0f); }
        }
    }
}

/// The reverse: [B, hp] back out to the caller's [.., B, H]. A length-0
/// sequence's final state is zeroed, per null_state_default_value.
template <typename T>
void store_state(TensorView dst, int64_t d, int64_t batch, int64_t hidden,
                 int64_t hp, const T* src, const int32_t* seq_len)
{
    if (dst.is_empty()) { return; }
    for (int64_t b = 0; b < batch; ++b) {
        T* out = dst.ptr<T>(static_cast<size_t>(d * batch + b));
        if (seq_len != nullptr && seq_len[b] == 0) {
            for (int64_t h = 0; h < hidden; ++h) { s_store(out + h, 0.0f); }
            continue;
        }
        std::memcpy(out, src + b * hp, static_cast<size_t>(hidden) * sizeof(T));
    }
}

// =========================================================================
//  The call, resolved once
// =========================================================================

/// Everything a direction reads and nothing it writes: the plan, the blob (when
/// the caller prepacked one), the operand views and the numbers derived from
/// their strides. Directions differ only in which rows of W, R and B they take.
template <typename T>
struct Run {
    RecurrentParams p;
    RecPlan plan;
    RecurrentPackedHeader q8_hdr;           ///< the header a per-call q8 pack writes
    const RecurrentPackedHeader* blob = nullptr;  ///< the caller's, when prepacked
    bool prepacked = false;

    TensorView y, yh, yc;
    const TensorView* x = nullptr;
    const TensorView* w = nullptr;
    const TensorView* r = nullptr;
    const TensorView* bias = nullptr;
    const TensorView* init_h = nullptr;
    const TensorView* init_c = nullptr;
    const int32_t* seq_len = nullptr;

    int64_t B = 0, I = 0, H = 0, D = 0, Tn = 0, G = 0;
    int64_t hp = 0;   ///< hidden padded to blocks of 8
    int64_t ldc = 0;  ///< the gate row stride
    bool q8 = false, is_gru = false, is_lstm = false, lbr = false;
    T* y_base = nullptr;  ///< Y's first element, null when Y is not written
    int64_t y_stride = 0;
    int64_t x_rows = 0;
    RowScales x_scales;
};

/// Resolve the plan and validate the blob. A blob packed for another geometry is
/// refused here rather than misread by the packers.
template <typename T>
Status prepare_run(const RecurrentParams& p, std::span<TensorView> outputs,
                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                   Run<T>& run)
{
    run.p = p;
    run.B = p.batch;
    run.I = p.input;
    run.H = p.hidden;
    run.D = p.directions;
    run.Tn = p.seq;
    run.G = recurrent_gates(p.mode);
    run.is_gru = p.mode == RecurrentMode::Gru;
    run.is_lstm = p.mode == RecurrentMode::Lstm;
    run.lbr = run.is_gru && p.linear_before_reset;

    NNOPS_TRY(build_plan(p, ctx, run.plan));
    run.hp = run.plan.hp;
    run.ldc = run.plan.total_cols;
    run.q8 = run.plan.q8;
    run.q8_hdr = packed_header(p);

    run.y = outputs[0];
    run.yh = outputs[1];
    run.yc = outputs.size() > 2 ? outputs[2] : TensorView{};

    run.x = &inputs[0];
    run.w = &inputs[1];
    run.r = &inputs[2];
    run.bias = (inputs.size() > 3 && !inputs[3].is_empty()) ? &inputs[3] : nullptr;
    run.init_h = (inputs.size() > 5 && !inputs[5].is_empty()) ? &inputs[5] : nullptr;
    run.init_c = (inputs.size() > 6 && !inputs[6].is_empty()) ? &inputs[6] : nullptr;
    run.seq_len = (inputs.size() > 4 && !inputs[4].is_empty())
                      ? inputs[4].ptr<int32_t>() : nullptr;

    run.y_base = run.y.is_empty() ? nullptr : run.y.ptr<T>();
    run.y_stride = run.y.is_empty() ? 0 : run.y.row_stride_elems();
    run.x_rows = run.x->row_stride_elems();
    // An s8 activation carries its own per-row table; a floating one is
    // quantized inside the projection from each row's max.
    if (run.q8 && run.x->data_type() == DataType::s8) {
        run.x_scales = row_scales_of(run.x->quant_params(), run.Tn * run.B);
    }

    // A prepacked W is every weight this driver would build — W and R panels and
    // the bias vectors — so the packing scratch disappears with it.
    if (is_packed_recurrent(run.w->layout())) {
        const int64_t esz = static_cast<int64_t>(sizeof(T));
        const char* blob = reinterpret_cast<const char*>(run.w->ptr<T>());
        const auto* h = reinterpret_cast<const RecurrentPackedHeader*>(blob);
        if (h->magic != kRecurrentPackedMagic || h->version != kRecurrentPackedVersion ||
            h->elem_size != (run.q8 ? int64_t{1} : esz) ||
            h->q8 != (run.q8 ? int64_t{1} : int64_t{0}) ||
            h->input != run.I || h->hidden != run.H ||
            h->directions != run.D || h->gates != run.G ||
            h->lbr != static_cast<int64_t>(run.lbr ? 1 : 0) ||
            run.w->numel() < recurrent_packed_numel(p)) {
            return Status::invalid_argument;
        }
        run.blob = h;
        run.prepacked = true;
    }
    return Status::ok;
}

/// The per-call scratch, in T elements and in bytes (the s8 panels are one byte
/// per element). Zero for a prepacked weight.
struct ScratchSize {
    int64_t elems = 0;   ///< in T
    int64_t qbytes = 0;  ///< byte-sized regions
};

template <typename T>
ScratchSize scratch_size(const Run<T>& run)
{
    const RecPlan& plan = run.plan;
    const bool q8 = run.q8;
    const bool lbr = run.lbr;

    ScratchSize s;
    if (!run.prepacked) {
        if (q8) {
            s.qbytes += plan.total_cols * std::max(run.I, run.H);  // interleave staging
            s.qbytes += q8_packed_bytes(run.I, plan.total_cols);
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                s.qbytes += q8_packed_bytes(run.H, plan.phase[ph].cols);
            }
            s.qbytes += 4 * plan.total_cols;                       // W's scales
            s.qbytes += 8 * run.G * run.H;                         // bias staging, f32
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                s.qbytes += 4 * plan.phase[ph].cols;               // R's scales
                s.qbytes += 4 * plan.phase[ph].cols;               // bias, f32
            }
            if (lbr) { s.qbytes += 4 * plan.total_cols; }          // Rbh, f32
        } else {
            s.elems += plan.total_cols * std::max(run.I, run.H);   // interleave staging
            s.elems += packed_b_numel(plan.total_cols, plan.wgeo.total);
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                const PhaseGeom& g = plan.phase[ph];
                s.elems += packed_b_numel(g.cols, g.rgeo.total);
                s.elems += g.cols;                                 // bias
            }
            if (lbr) { s.elems += plan.phase[0].cols; }            // Rbh, kept apart
        }
    }
    if (q8) {
        // The blob stores the bias as f32 whatever the compute dtype, so a q8
        // direction converts it once.
        for (int64_t ph = 0; ph < plan.nphases; ++ph) { s.elems += plan.phase[ph].cols; }
        if (lbr) { s.elems += plan.total_cols; }
        s.qbytes += plan.slots * (plan.a_bytes + plan.c_elems * 4 + plan.s_elems * 4);
    }
    s.elems += plan.chunk * run.B * run.ldc;   // the input projection
    if (lbr) { s.elems += run.B * run.ldc; }   // h~'s second accumulation
    s.elems += 2 * run.B * run.hp;             // h_prev, h_next
    if (run.is_lstm) { s.elems += 2 * run.B * run.hp; }        // c_prev, c_next
    if (run.is_gru && !lbr) { s.elems += 2 * run.B * run.hp; }  // z, r⊙H
    return s;
}

// =========================================================================
//  One direction's weights
// =========================================================================

/// The packed panels and bias vectors a step reads, plus the quantized route's
/// s8 slabs and f32 scales. Only the half the route uses is filled.
template <typename T>
struct DirWeights {
    const T* w = nullptr;
    const T* r[2] = {nullptr, nullptr};
    const T* bias[2] = {nullptr, nullptr};
    const T* rh = nullptr;   ///< GRU h gate's Rbh, linear_before_reset only

    const int8_t* q_w = nullptr;
    const int8_t* q_r[2] = {nullptr, nullptr};
    const float* q_sw = nullptr;
    const float* q_sr[2] = {nullptr, nullptr};
    const float* q_bias[2] = {nullptr, nullptr};
    const float* q_rh = nullptr;
};

/// Point at direction d's regions of the caller's blob — nothing is copied.
template <typename T>
void weights_from_blob(const Run<T>& run, int64_t d, DirWeights<T>& dw)
{
    const RecurrentPackedHeader* h = run.blob;
    const PackedWeightOffsets o = packed_weight_offsets(*h);
    const char* base =
        reinterpret_cast<const char*>(h) + packed_header_bytes() + d * h->dir_bytes;

    if (run.q8) {
        dw.q_w = reinterpret_cast<const int8_t*>(base + o.w);
        dw.q_sw = reinterpret_cast<const float*>(base + o.sw);
        for (int64_t ph = 0; ph < run.plan.nphases; ++ph) {
            dw.q_r[ph] = reinterpret_cast<const int8_t*>(base + o.r[ph]);
            dw.q_sr[ph] = reinterpret_cast<const float*>(base + o.sr[ph]);
            dw.q_bias[ph] = reinterpret_cast<const float*>(base + o.bias[ph]);
        }
        dw.q_rh = h->lbr != 0 ? reinterpret_cast<const float*>(base + o.rh) : nullptr;
        return;
    }

    dw.w = reinterpret_cast<const T*>(base + o.w);
    for (int64_t ph = 0; ph < run.plan.nphases; ++ph) {
        dw.r[ph] = reinterpret_cast<const T*>(base + o.r[ph]);
        dw.bias[ph] = reinterpret_cast<const T*>(base + o.bias[ph]);
    }
    dw.rh = h->lbr != 0 ? reinterpret_cast<const T*>(base + o.rh) : nullptr;
}

/// The blob keeps the bias as f32 (a pack cannot know the compute dtype); the
/// epilogue reads it in T.
template <typename T>
void convert_q8_bias(const Run<T>& run, Arena& arena, DirWeights<T>& dw)
{
    for (int64_t ph = 0; ph < run.plan.nphases; ++ph) {
        const int64_t cols = run.plan.phase[ph].cols;
        T* bp = arena.take<T>(cols);
        for (int64_t i = 0; i < cols; ++i) { s_store(bp + i, dw.q_bias[ph][i]); }
        dw.bias[ph] = bp;
    }
    if (run.lbr) {
        T* rp = arena.take<T>(run.plan.total_cols);
        for (int64_t i = 0; i < run.plan.total_cols; ++i) { s_store(rp + i, dw.q_rh[i]); }
        dw.rh = rp;
    }
}

/// Interleave, pack and scale direction d's s8 weights into scratch, then
/// convert the bias to the compute dtype.
template <typename T>
void pack_q8_weights(const Run<T>& run, int64_t d, Arena& arena, DirWeights<T>& dw)
{
    const RecPlan& plan = run.plan;
    int8_t* raw8 = reinterpret_cast<int8_t*>(
        arena.take_bytes(plan.total_cols * std::max(run.I, run.H)));
    Q8Regions reg;
    reg.w = reinterpret_cast<int8_t*>(
        arena.take_bytes(q8_packed_bytes(run.I, plan.total_cols)));
    reg.sw = reinterpret_cast<float*>(arena.take_bytes(4 * plan.total_cols));
    for (int64_t ph = 0; ph < plan.nphases; ++ph) {
        reg.r[ph] = reinterpret_cast<int8_t*>(
            arena.take_bytes(q8_packed_bytes(run.H, plan.phase[ph].cols)));
        reg.sr[ph] = reinterpret_cast<float*>(
            arena.take_bytes(4 * plan.phase[ph].cols));
        reg.bias[ph] = reinterpret_cast<float*>(
            arena.take_bytes(4 * plan.phase[ph].cols));
    }
    if (run.lbr) {
        reg.rh = reinterpret_cast<float*>(arena.take_bytes(4 * plan.total_cols));
    }
    float* bias_f32 = reinterpret_cast<float*>(arena.take_bytes(8 * run.G * run.H));
    prepare_q8_direction(run.p, run.q8_hdr, d, *run.w, *run.r, run.bias, raw8,
                         bias_f32, reg);
    dw.q_w = reg.w;
    dw.q_sw = reg.sw;
    for (int64_t ph = 0; ph < plan.nphases; ++ph) {
        dw.q_r[ph] = reg.r[ph];
        dw.q_sr[ph] = reg.sr[ph];
        dw.q_bias[ph] = reg.bias[ph];
    }
    dw.q_rh = reg.rh;
    convert_q8_bias(run, arena, dw);
}

/// Interleave and pack direction d's fp weights into scratch: W as one
/// [total_cols, I] matrix, then each phase's R, then the bias vectors.
template <typename T>
void pack_fp_weights(const Run<T>& run, int64_t d, Arena& arena,
                     const ComputeContext& ctx, DirWeights<T>& dw)
{
    const RecPlan& plan = run.plan;
    const int64_t esz = static_cast<int64_t>(sizeof(T));
    T* raw = arena.take<T>(plan.total_cols * std::max(run.I, run.H));

    // W: every phase's gate rows into one matrix, then one pack.
    for (int64_t ph = 0; ph < plan.nphases; ++ph) {
        const PhaseGeom& g = plan.phase[ph];
        interleave_gates(run.w->ptr<T>(static_cast<size_t>(d * run.G * run.H)),
                         run.w->row_stride_elems(), run.H, run.I, g.gates,
                         phase_gate_offset(run.p.mode, run.lbr, ph),
                         raw + plan.col_base[ph] * run.I);
    }
    {
        int64_t sh[2] = {plan.total_cols, run.I};
        TensorView rv(std::span<const int64_t>(sh, 2), run.p.dtype, raw, run.I * esz);
        TensorView pv = packed_view(
            plan.total_cols, plan.wgeo.total,
            arena.take<T>(packed_b_numel(plan.total_cols, plan.wgeo.total)));
        matmul_packed_b_pack(rv, pv, /*transpose_b=*/true, ctx);
        dw.w = pv.ptr<T>();
    }

    // The bias is [D, 2·G·H], so direction d is row d — `ptr` counts rows, and
    // each row is already the whole of one direction.
    const T* wb = run.bias ? run.bias->ptr<T>(static_cast<size_t>(d)) : nullptr;
    const T* rb = wb ? wb + run.G * run.H : nullptr;

    for (int64_t ph = 0; ph < plan.nphases; ++ph) {
        const PhaseGeom& g = plan.phase[ph];
        const int64_t gate_off = phase_gate_offset(run.p.mode, run.lbr, ph);

        interleave_gates(run.r->ptr<T>(static_cast<size_t>(d * run.G * run.H)),
                         run.r->row_stride_elems(), run.H, run.H, g.gates, gate_off, raw);
        {
            int64_t sh[2] = {g.cols, run.H};
            TensorView rv(std::span<const int64_t>(sh, 2), run.p.dtype, raw, run.H * esz);
            TensorView pv = packed_view(
                g.cols, g.rgeo.total,
                arena.take<T>(packed_b_numel(g.cols, g.rgeo.total)));
            matmul_packed_b_pack(rv, pv, /*transpose_b=*/true, ctx);
            dw.r[ph] = pv.ptr<T>();
        }

        // Rbh belongs after the recurrent product under linear_before_reset,
        // so the h gate's bias vector takes Wb alone here. The h gate is the
        // third of a merged GRU phase and the only one of a h~ phase.
        const int64_t h_gate = (g.kind == PhaseKind::GruZr) ? -1
                             : (g.kind == PhaseKind::GruAll) ? 2 : 0;
        const int64_t all_gates = (int64_t{1} << g.gates) - 1;
        const int64_t h_bit = (h_gate < 0) ? 0 : (int64_t{1} << h_gate);
        T* bp = arena.take<T>(g.cols);
        build_bias(wb, rb, run.H, g.gates,
                   run.lbr ? (all_gates & ~h_bit) : all_gates, gate_off, bp);
        dw.bias[ph] = bp;

        if (run.lbr) {
            T* rbp = arena.take<T>(g.cols);
            build_bias(static_cast<const T*>(nullptr), rb, run.H, g.gates,
                       h_bit, gate_off, rbp);
            dw.rh = rbp;
        }
    }
}

// =========================================================================
//  One direction's timeline
// =========================================================================

/// One direction's live state: the step buffers and the gate-row slices of the
/// projection and the linear_before_reset accumulation.
template <typename T>
struct DirState {
    T* in_proj = nullptr;
    T* rec = nullptr;  ///< h~'s second accumulation, linear_before_reset only
    T* h_a = nullptr;
    T* h_b = nullptr;
    T* c_a = nullptr;
    T* c_b = nullptr;
    T* zb = nullptr;   ///< GRU z, the h~ phase's blend
    T* aux = nullptr;  ///< GRU r⊙Ht-1, the h~ phase's recurrent input
};

/// The step kernel's view of one phase at one timestep.
template <typename T>
StepCtx<T> step_ctx(const Run<T>& run, const DirState<T>& ds,
                    const DirWeights<T>& dw, const Q8Scratch& qs,
                    const PhaseGeom& g, int64_t ph, int64_t d, int64_t t,
                    const T* ip_row)
{
    const bool h_gate = g.kind == PhaseKind::GruH;

    StepCtx<T> s;
    s.batch = run.B;
    s.hidden = run.H;
    s.hp = run.hp;
    s.cols = g.cols;
    s.kc = run.plan.kc;
    s.ldc = run.ldc;
    s.m_blocks = g.m_blocks;
    s.n_blocks = g.n_blocks;
    s.mc = g.mc;
    s.nc = g.nc;
    s.a = h_gate ? ds.aux : ds.h_a;
    s.in_proj = const_cast<T*>(ip_row) + run.plan.col_base[ph];
    s.rec = run.lbr ? ds.rec + run.plan.col_base[ph] : nullptr;
    s.packed_r = dw.r[ph];
    s.r_ldd = g.rgeo.ldd;
    s.r_nr = g.rgeo.nr;
    s.q_r = dw.q_r[ph];
    s.sr = dw.q_sr[ph];
    s.qs = qs;
    s.bias = dw.bias[ph];
    s.rh_bias = (run.lbr && g.kind == PhaseKind::GruAll) ? dw.rh : nullptr;
    s.h_prev = ds.h_a;
    s.h_next = ds.h_b;
    s.c_prev = run.is_lstm ? ds.c_a : nullptr;
    s.c_next = run.is_lstm ? ds.c_b : nullptr;
    s.zb = ds.zb;
    s.aux = ds.aux;
    // Row (t, d)'s first element: B rows of y_stride elements each.
    s.y_row0 = run.y_base
                   ? run.y_base + ((t * run.D + d) * run.B) * run.y_stride
                   : nullptr;
    s.y_stride = run.y_stride;
    s.seq_len = run.seq_len;
    s.t = t;
    s.clip = run.p.clip;
    s.act_f = run.p.act_f;
    s.act_g = run.p.act_g;
    s.act_h = run.p.act_h;
    s.alpha_f = run.p.alpha_f;
    s.alpha_g = run.p.alpha_g;
    s.alpha_h = run.p.alpha_h;
    s.input_forget = run.p.input_forget;
    s.linear_before_reset = run.lbr;
    return s;
}

/// Every phase of one timestep, in order: the fp GRU's h~ phase reads what its
/// z/r phase left behind.
template <typename T>
void run_step(const Run<T>& run, const DirState<T>& ds, const DirWeights<T>& dw,
              const Q8Scratch& qs, int64_t d, int64_t t, const T* ip_row,
              const ComputeContext& ctx)
{
    for (int64_t ph = 0; ph < run.plan.nphases; ++ph) {
        const PhaseGeom& g = run.plan.phase[ph];
        StepCtx<T> s = step_ctx(run, ds, dw, qs, g, ph, d, t, ip_row);
        if (run.q8) {
            run_phase_q8(s, g.kind, ctx);
        } else {
            run_phase(s, g.kind, ctx);
        }
    }
}

/// Pack direction d's weights, then walk its chunks and timesteps. The step
/// order is forward ascending, reverse descending from the tail chunk, so both
/// directions write Y in timeline order.
template <typename T>
void run_direction(const Run<T>& run, int64_t d, void* base, size_t bytes,
                   const ComputeContext& ctx)
{
    Arena arena(base, bytes);

    DirState<T> ds;
    ds.in_proj = arena.take<T>(run.plan.chunk * run.B * run.ldc);
    ds.rec = run.lbr ? arena.take<T>(run.B * run.ldc) : nullptr;
    ds.h_a = arena.take<T>(run.B * run.hp);
    ds.h_b = arena.take<T>(run.B * run.hp);
    if (run.is_lstm) {
        ds.c_a = arena.take<T>(run.B * run.hp);
        ds.c_b = arena.take<T>(run.B * run.hp);
    }
    if (run.is_gru && !run.lbr) {
        ds.zb = arena.take<T>(run.B * run.hp);
        ds.aux = arena.take<T>(run.B * run.hp);
    }

    Q8Scratch qs;
    if (run.q8) {
        qs.slots = run.plan.slots;
        qs.a_bytes = run.plan.a_bytes;
        qs.c_elems = run.plan.c_elems;
        qs.s_elems = run.plan.s_elems;
        qs.a = reinterpret_cast<int8_t*>(
            arena.take_bytes(run.plan.slots * run.plan.a_bytes));
        qs.c = reinterpret_cast<int32_t*>(
            arena.take_bytes(run.plan.slots * run.plan.c_elems * 4));
        qs.s = reinterpret_cast<float*>(
            arena.take_bytes(run.plan.slots * run.plan.s_elems * 4));
    }

    DirWeights<T> dw;
    if (run.prepacked) {
        weights_from_blob(run, d, dw);
        if (run.q8) { convert_q8_bias(run, arena, dw); }
    } else if (run.q8) {
        pack_q8_weights(run, d, arena, dw);
    } else {
        pack_fp_weights(run, d, arena, ctx, dw);
    }

    load_state(run.init_h, d, run.B, run.H, run.hp, ds.h_a);
    if (run.is_lstm) { load_state(run.init_c, d, run.B, run.H, run.hp, ds.c_a); }

    const bool fwd =
        (run.D > 1) ? (d == 0) : (run.p.direction != RnnDirection::Reverse);
    const int64_t nchunks =
        run.Tn > 0 ? (run.Tn + run.plan.chunk - 1) / run.plan.chunk : 0;

    for (int64_t ch = 0; ch < nchunks; ++ch) {
        const int64_t t0 = fwd
            ? ch * run.plan.chunk
            : std::max<int64_t>(0, run.Tn - (ch + 1) * run.plan.chunk);
        const int64_t nrows = std::min<int64_t>(run.plan.chunk, run.Tn - t0);
        if (nrows <= 0) { continue; }

        // One GEMM over the whole gate width: both phases of a chunk read the
        // same X rows, and neither's input projection waits on the other's
        // recurrence.
        if (run.q8) {
            project_chunk_q8<T>(*run.x, t0 * run.B, ds.in_proj, run.ldc,
                                dw.q_w, dw.q_sw, run.plan.total_cols, run.I,
                                nrows * run.B, run.plan.proj_mc, run.plan.proj_nc,
                                run.x_scales, qs, ctx);
        } else {
            project_chunk<T>(run.x->ptr<T>(static_cast<size_t>(t0 * run.B)),
                             run.x_rows, ds.in_proj, run.ldc, dw.w,
                             run.plan.wgeo.ldd, run.plan.wgeo.nr, nrows * run.B,
                             run.plan.total_cols, run.I, run.plan.kc,
                             run.plan.total_cols, ctx);
        }

        const int64_t t_step = fwd ? 1 : -1;
        const int64_t t_first = fwd ? t0 : t0 + nrows - 1;

        for (int64_t i = 0; i < nrows; ++i) {
            const int64_t t = t_first + i * t_step;
            const T* ip_row = ds.in_proj + ((t - t0) * run.B) * run.ldc;
            run_step(run, ds, dw, qs, d, t, ip_row, ctx);
            std::swap(ds.h_a, ds.h_b);
            if (run.is_lstm) { std::swap(ds.c_a, ds.c_b); }
        }
    }

    store_state(run.yh, d, run.B, run.H, run.hp, ds.h_a, run.seq_len);
    if (run.is_lstm) { store_state(run.yc, d, run.B, run.H, run.hp, ds.c_a, run.seq_len); }
}

}  // namespace

// =========================================================================
//  Driver
// =========================================================================

template <typename T>
Status recurrent_impl(const RecurrentParams& p, std::span<TensorView> outputs,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    if (p.batch <= 0 || p.input <= 0 || p.hidden <= 0 || p.directions <= 0) {
        return Status::ok;
    }

    Run<T> run;
    NNOPS_TRY(prepare_run(p, outputs, inputs, ctx, run));

    const ScratchSize sz = scratch_size(run);
    const size_t bytes = static_cast<size_t>(sz.elems) * sizeof(T) +
                         static_cast<size_t>(sz.qbytes) + 4096;
    PoolPtr owner(bytes);
    if (!owner) { return Status::out_of_memory; }

    for (int64_t d = 0; d < run.D; ++d) {
        run_direction(run, d, owner.get(), bytes, ctx);
    }
    return Status::ok;
}

Status recurrent_cpu(const RecurrentParams& p,
                     std::span<TensorView> outputs,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx)
{
    if (p.dtype != DataType::f32 && p.dtype != DataType::f16) {
        return Status::unsupported;
    }
    return p.dtype == DataType::f32
               ? recurrent_impl<float>(p, outputs, inputs, ctx)
               : recurrent_impl<half>(p, outputs, inputs, ctx);
}

}  // namespace nnops::backend::cpu
