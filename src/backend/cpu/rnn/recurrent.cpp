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
/// file walks the timeline: chunk the sequence, project, step, swap.

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
void store_state(TensorView& dst, int64_t d, int64_t batch, int64_t hidden,
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

}  // namespace

// =========================================================================
//  Driver
// =========================================================================

template <typename T>
Status recurrent_impl(const RecurrentParams& p, std::span<TensorView> outputs,
                      std::span<const TensorView> inputs, const ComputeContext& ctx)
{
    const int64_t B = p.batch;
    const int64_t I = p.input;
    const int64_t H = p.hidden;
    const int64_t D = p.directions;
    const int64_t Tn = p.seq;
    const int64_t esz = static_cast<int64_t>(sizeof(T));
    const int64_t G = recurrent_gates(p.mode);
    const bool is_gru = p.mode == RecurrentMode::Gru;
    const bool is_lstm = p.mode == RecurrentMode::Lstm;
    const bool lbr = is_gru && p.linear_before_reset;

    if (B <= 0 || I <= 0 || H <= 0 || D <= 0) { return Status::ok; }

    RecPlan plan;
    NNOPS_TRY(build_plan(p, ctx, plan));
    const int64_t hp = plan.hp;
    const int64_t ldc = plan.total_cols;
    const DataType dt = p.dtype;
    const bool q8 = plan.q8;
    // The header a per-call q8 pack writes, for the geometry the blob uses too.
    const RecurrentPackedHeader q8_hdr = packed_header(p);

    // A prepacked W is every weight this driver would build — W and R panels and
    // the bias vectors — so the packing scratch disappears with it.
    const TensorView& w_in = inputs[1];
    const bool prepacked = is_packed_recurrent(w_in.layout());
    const RecurrentPackedHeader* ph_hdr = nullptr;
    if (prepacked) {
        const char* blob = reinterpret_cast<const char*>(w_in.ptr<T>());
        const auto* h = reinterpret_cast<const RecurrentPackedHeader*>(blob);
        if (h->magic != kRecurrentPackedMagic || h->version != kRecurrentPackedVersion ||
            h->elem_size != (q8 ? int64_t{1} : esz) ||
            h->q8 != (q8 ? int64_t{1} : int64_t{0}) ||
            h->input != I || h->hidden != H ||
            h->directions != D || h->gates != G ||
            h->lbr != static_cast<int64_t>(lbr ? 1 : 0) ||
            w_in.numel() < recurrent_packed_numel(p)) {
            return Status::invalid_argument;
        }
        ph_hdr = h;
    }

    // ---- scratch: one pooled block, carved per direction ------------------
    int64_t elems = 0;   // in T
    int64_t qbytes = 0;  // the s8 panels are one byte per element
    if (!prepacked) {
        if (q8) {
            qbytes += plan.total_cols * std::max(I, H);  // interleave staging
            qbytes += q8_packed_bytes(I, plan.total_cols);
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                qbytes += q8_packed_bytes(H, plan.phase[ph].cols);
            }
            qbytes += 4 * plan.total_cols;                       // W's scales
            qbytes += 8 * G * H;                                 // bias staging, f32
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                qbytes += 4 * plan.phase[ph].cols;               // R's scales
                qbytes += 4 * plan.phase[ph].cols;               // bias, f32
            }
            if (lbr) { qbytes += 4 * plan.total_cols; }          // Rbh, f32
        } else {
            elems += plan.total_cols * std::max(I, H);  // the interleave staging area
            elems += packed_b_numel(plan.total_cols, plan.wgeo.total);
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                const PhaseGeom& g = plan.phase[ph];
                elems += packed_b_numel(g.cols, g.rgeo.total);
                elems += g.cols;                        // bias
            }
            if (lbr) { elems += plan.phase[0].cols; }   // Rbh, kept apart
        }
    }
    if (q8) {
        // The blob stores the bias as f32 whatever the compute dtype, so a q8
        // direction converts it once.
        for (int64_t ph = 0; ph < plan.nphases; ++ph) { elems += plan.phase[ph].cols; }
        if (lbr) { elems += plan.total_cols; }
        qbytes += plan.slots * (plan.a_bytes + plan.c_elems * 4 + plan.s_elems * 4);
    }
    elems += plan.chunk * B * ldc;              // the input projection
    if (lbr) { elems += B * ldc; }              // h~'s second accumulation
    elems += 2 * B * hp;                        // h_prev, h_next
    if (is_lstm) { elems += 2 * B * hp; }       // c_prev, c_next
    if (is_gru && !lbr) { elems += 2 * B * hp; }  // z, r⊙H

    const size_t bytes = static_cast<size_t>(elems) * sizeof(T) +
                         static_cast<size_t>(qbytes) + 4096;
    PoolPtr owner(bytes);
    if (!owner) { return Status::out_of_memory; }

    TensorView& y = outputs[0];
    TensorView& yh = outputs[1];
    TensorView dummy;
    TensorView& yc = outputs.size() > 2 ? outputs[2] : dummy;

    const TensorView& x = inputs[0];
    const TensorView& w = inputs[1];
    const TensorView& r = inputs[2];
    const TensorView* bias =
        (inputs.size() > 3 && !inputs[3].is_empty()) ? &inputs[3] : nullptr;
    const TensorView* init_h =
        (inputs.size() > 5 && !inputs[5].is_empty()) ? &inputs[5] : nullptr;
    const TensorView* init_c =
        (inputs.size() > 6 && !inputs[6].is_empty()) ? &inputs[6] : nullptr;
    const int32_t* seq_len = (inputs.size() > 4 && !inputs[4].is_empty())
                                 ? inputs[4].ptr<int32_t>() : nullptr;

    const int64_t y_stride = y.is_empty() ? 0 : y.row_stride_elems();
    const int64_t x_rows = x.row_stride_elems();
    // An s8 activation carries its own per-row table; a floating one is
    // quantized inside the projection from each row's max.
    RowScales x_scales;
    if (q8 && x.data_type() == DataType::s8) {
        x_scales = row_scales_of(x.quant_params(), Tn * B);
    }

    for (int64_t d = 0; d < D; ++d) {
        Arena arena(owner.get(), bytes);

        T* in_proj = arena.take<T>(plan.chunk * B * ldc);
        T* rec = lbr ? arena.take<T>(B * ldc) : nullptr;
        T* h_a = arena.take<T>(B * hp);
        T* h_b = arena.take<T>(B * hp);
        T* c_a = is_lstm ? arena.take<T>(B * hp) : nullptr;
        T* c_b = is_lstm ? arena.take<T>(B * hp) : nullptr;
        T* zb = (is_gru && !lbr) ? arena.take<T>(B * hp) : nullptr;
        T* aux = (is_gru && !lbr) ? arena.take<T>(B * hp) : nullptr;
        const T* packed_w = nullptr;
        const T* packed_r[2] = {nullptr, nullptr};
        const T* bias_p[2] = {nullptr, nullptr};
        const T* rh_bias = nullptr;

        // ---- the quantized route's own scratch -----------------------------
        Q8Scratch qs;
        const int8_t* q_w = nullptr;
        const int8_t* q_r[2] = {nullptr, nullptr};
        const float* q_sw = nullptr;
        const float* q_sr[2] = {nullptr, nullptr};
        const float* q_bias[2] = {nullptr, nullptr};
        const float* q_rh = nullptr;
        if (q8) {
            qs.slots = plan.slots;
            qs.a_bytes = plan.a_bytes;
            qs.c_elems = plan.c_elems;
            qs.s_elems = plan.s_elems;
            qs.a = reinterpret_cast<int8_t*>(arena.take_bytes(plan.slots * plan.a_bytes));
            qs.c = reinterpret_cast<int32_t*>(
                arena.take_bytes(plan.slots * plan.c_elems * 4));
            qs.s = reinterpret_cast<float*>(
                arena.take_bytes(plan.slots * plan.s_elems * 4));

            if (prepacked) {
                const PackedWeightOffsets o = packed_weight_offsets(*ph_hdr);
                const char* base =
                    reinterpret_cast<const char*>(ph_hdr) + packed_header_bytes() +
                    d * ph_hdr->dir_bytes;
                q_w = reinterpret_cast<const int8_t*>(base + o.w);
                q_sw = reinterpret_cast<const float*>(base + o.sw);
                for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                    q_r[ph] = reinterpret_cast<const int8_t*>(base + o.r[ph]);
                    q_sr[ph] = reinterpret_cast<const float*>(base + o.sr[ph]);
                    q_bias[ph] = reinterpret_cast<const float*>(base + o.bias[ph]);
                }
                q_rh = ph_hdr->lbr != 0 ? reinterpret_cast<const float*>(base + o.rh)
                                        : nullptr;
            } else {
                int8_t* raw8 = reinterpret_cast<int8_t*>(
                    arena.take_bytes(plan.total_cols * std::max(I, H)));
                Q8Regions reg;
                reg.w = reinterpret_cast<int8_t*>(
                    arena.take_bytes(q8_packed_bytes(I, plan.total_cols)));
                reg.sw = reinterpret_cast<float*>(
                    arena.take_bytes(4 * plan.total_cols));
                for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                    reg.r[ph] = reinterpret_cast<int8_t*>(
                        arena.take_bytes(q8_packed_bytes(H, plan.phase[ph].cols)));
                    reg.sr[ph] = reinterpret_cast<float*>(
                        arena.take_bytes(4 * plan.phase[ph].cols));
                    reg.bias[ph] = reinterpret_cast<float*>(
                        arena.take_bytes(4 * plan.phase[ph].cols));
                }
                if (lbr) {
                    reg.rh = reinterpret_cast<float*>(arena.take_bytes(4 * plan.total_cols));
                }
                float* bias_f32 = reinterpret_cast<float*>(
                    arena.take_bytes(8 * G * H));
                prepare_q8_direction(p, q8_hdr, d, w_in, r, bias, raw8, bias_f32,
                                     reg);
                q_w = reg.w;
                q_sw = reg.sw;
                for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                    q_r[ph] = reg.r[ph];
                    q_sr[ph] = reg.sr[ph];
                    q_bias[ph] = reg.bias[ph];
                }
                q_rh = reg.rh;
            }

            // The blob keeps the bias as f32 (a pack cannot know the compute
            // dtype); the epilogue reads it in T.
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                const int64_t cols = plan.phase[ph].cols;
                T* bp = arena.take<T>(cols);
                for (int64_t i = 0; i < cols; ++i) { s_store(bp + i, q_bias[ph][i]); }
                bias_p[ph] = bp;
            }
            if (lbr) {
                T* rp = arena.take<T>(plan.total_cols);
                for (int64_t i = 0; i < plan.total_cols; ++i) { s_store(rp + i, q_rh[i]); }
                rh_bias = rp;
            }
        } else if (prepacked) {
            const PackedWeightOffsets o = packed_weight_offsets(*ph_hdr);
            const char* base =
                reinterpret_cast<const char*>(ph_hdr) + packed_header_bytes() +
                d * ph_hdr->dir_bytes;
            packed_w = reinterpret_cast<const T*>(base + o.w);
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                packed_r[ph] = reinterpret_cast<const T*>(base + o.r[ph]);
                bias_p[ph] = reinterpret_cast<const T*>(base + o.bias[ph]);
            }
            rh_bias = ph_hdr->lbr != 0 ? reinterpret_cast<const T*>(base + o.rh)
                                       : nullptr;
        } else {
            T* raw = arena.take<T>(plan.total_cols * std::max(I, H));

            // ---- the direction's weights, interleaved and packed once -------
            // W: every phase's gate rows into one matrix, then one pack.
            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                const PhaseGeom& g = plan.phase[ph];
                interleave_gates(w_in.ptr<T>(static_cast<size_t>(d * G * H)),
                                 w_in.row_stride_elems(), H, I, g.gates,
                                 phase_gate_offset(p.mode, lbr, ph),
                                 raw + plan.col_base[ph] * I);
            }
            {
                int64_t sh[2] = {plan.total_cols, I};
                TensorView rv(std::span<const int64_t>(sh, 2), dt, raw, I * esz);
                TensorView pv = packed_view(
                    plan.total_cols, plan.wgeo.total,
                    arena.take<T>(packed_b_numel(plan.total_cols, plan.wgeo.total)));
                matmul_packed_b_pack(rv, pv, /*transpose_b=*/true, ctx);
                packed_w = pv.ptr<T>();
            }

            // The bias is [D, 2·G·H], so direction d is row d — `ptr` counts
            // rows, and each row is already the whole of one direction.
            const T* wb = bias ? bias->ptr<T>(static_cast<size_t>(d)) : nullptr;
            const T* rb = wb ? wb + G * H : nullptr;

            for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                const PhaseGeom& g = plan.phase[ph];
                const int64_t gate_off = phase_gate_offset(p.mode, lbr, ph);

                interleave_gates(r.ptr<T>(static_cast<size_t>(d * G * H)),
                                 r.row_stride_elems(), H, H, g.gates, gate_off, raw);
                {
                    int64_t sh[2] = {g.cols, H};
                    TensorView rv(std::span<const int64_t>(sh, 2), dt, raw, H * esz);
                    TensorView pv = packed_view(
                        g.cols, g.rgeo.total,
                        arena.take<T>(packed_b_numel(g.cols, g.rgeo.total)));
                    matmul_packed_b_pack(rv, pv, /*transpose_b=*/true, ctx);
                    packed_r[ph] = pv.ptr<T>();
                }

                // Rbh belongs after the recurrent product under linear_before_reset,
                // so the h gate's bias vector takes Wb alone here. The h gate is the
                // third of a merged GRU phase and the only one of a h~ phase.
                const int64_t h_gate = (g.kind == PhaseKind::GruZr) ? -1
                                     : (g.kind == PhaseKind::GruAll) ? 2 : 0;
                const int64_t all_gates = (int64_t{1} << g.gates) - 1;
                const int64_t h_bit = (h_gate < 0) ? 0 : (int64_t{1} << h_gate);
                T* bp = arena.take<T>(g.cols);
                build_bias(wb, rb, H, g.gates,
                           lbr ? (all_gates & ~h_bit) : all_gates, gate_off, bp);
                bias_p[ph] = bp;

                if (lbr) {
                    T* rbp = arena.take<T>(g.cols);
                    build_bias(static_cast<const T*>(nullptr), rb, H, g.gates,
                               h_bit, gate_off, rbp);
                    rh_bias = rbp;
                }
            }
        }

        load_state(init_h, d, B, H, hp, h_a);
        if (is_lstm) { load_state(init_c, d, B, H, hp, c_a); }

        // ---- walk the timeline: forward ascends, reverse descends ---------
        const bool fwd = (D > 1) ? (d == 0) : (p.direction != RnnDirection::Reverse);
        const int64_t nchunks = Tn > 0 ? (Tn + plan.chunk - 1) / plan.chunk : 0;

        for (int64_t ch = 0; ch < nchunks; ++ch) {
            const int64_t t0 = fwd
                ? ch * plan.chunk
                : std::max<int64_t>(0, Tn - (ch + 1) * plan.chunk);
            const int64_t nrows = std::min<int64_t>(plan.chunk, Tn - t0);
            if (nrows <= 0) { continue; }

            // One GEMM over the whole gate width: both phases of a chunk read
            // the same X rows, and neither's input projection waits on the
            // other's recurrence.
            if (q8) {
                project_chunk_q8<T>(x, t0 * B, in_proj, ldc, q_w, q_sw,
                                    plan.total_cols, I, nrows * B, plan.proj_mc,
                                    plan.proj_nc, x_scales, qs, ctx);
            } else {
                project_chunk<T>(x.ptr<T>(static_cast<size_t>(t0 * B)), x_rows,
                                 in_proj, ldc, packed_w,
                                 plan.wgeo.ldd, plan.wgeo.nr,
                                 nrows * B, plan.total_cols, I, plan.kc,
                                 plan.total_cols, ctx);
            }

            const int64_t t_step = fwd ? 1 : -1;
            const int64_t t_first = fwd ? t0 : t0 + nrows - 1;

            for (int64_t i = 0; i < nrows; ++i) {
                const int64_t t = t_first + i * t_step;
                const T* ip_row = in_proj + ((t - t0) * B) * ldc;

                for (int64_t ph = 0; ph < plan.nphases; ++ph) {
                    const PhaseGeom& g = plan.phase[ph];
                    const bool h_gate = g.kind == PhaseKind::GruH;

                    StepCtx<T> s;
                    s.batch = B;
                    s.hidden = H;
                    s.hp = hp;
                    s.cols = g.cols;
                    s.kc = plan.kc;
                    s.ldc = ldc;
                    s.m_blocks = g.m_blocks;
                    s.n_blocks = g.n_blocks;
                    s.mc = g.mc;
                    s.nc = g.nc;
                    s.a = h_gate ? aux : h_a;
                    s.in_proj = const_cast<T*>(ip_row) + plan.col_base[ph];
                    s.rec = lbr ? rec + plan.col_base[ph] : nullptr;
                    s.packed_r = packed_r[ph];
                    s.r_ldd = g.rgeo.ldd;
                    s.r_nr = g.rgeo.nr;
                    s.q_r = q_r[ph];
                    s.sr = q_sr[ph];
                    s.qs = qs;
                    s.bias = bias_p[ph];
                    s.rh_bias = (lbr && g.kind == PhaseKind::GruAll) ? rh_bias : nullptr;
                    s.h_prev = h_a;
                    s.h_next = h_b;
                    s.c_prev = is_lstm ? c_a : nullptr;
                    s.c_next = is_lstm ? c_b : nullptr;
                    s.zb = zb;
                    s.aux = aux;
                    s.y_row0 = y.is_empty()
                                   ? nullptr
                                   : y.ptr<T>(static_cast<size_t>((t * D + d) * B));
                    s.y_stride = y_stride;
                    s.seq_len = seq_len;
                    s.t = t;
                    s.clip = p.clip;
                    s.act_f = p.act_f;
                    s.act_g = p.act_g;
                    s.act_h = p.act_h;
                    s.alpha_f = p.alpha_f;
                    s.alpha_g = p.alpha_g;
                    s.alpha_h = p.alpha_h;
                    s.input_forget = p.input_forget;
                    s.linear_before_reset = lbr;
                    if (q8) {
                        run_phase_q8(s, g.kind, ctx);
                    } else {
                        run_phase(s, g.kind, ctx);
                    }
                }

                std::swap(h_a, h_b);
                if (is_lstm) { std::swap(c_a, c_b); }
            }
        }

        store_state(yh, d, B, H, hp, h_a, seq_len);
        if (is_lstm) { store_state(yc, d, B, H, hp, c_a, seq_len); }
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
