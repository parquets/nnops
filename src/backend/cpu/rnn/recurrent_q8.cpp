/// @file recurrent_q8.cpp
/// @brief The int8 twin of the recurrent family: weight preparation, the
///        chunked input projection and the per-step recurrent product.
///
/// W and R go in as int8 panels with interleaved f32 per-column scales; both
/// projections quantize A per row from its own max and dequantize the int32
/// accumulator into the fp buffer the shared epilogue reads. The per-column
/// scale of a packed panel is the scale of the gate row its column came from,
/// which is what lets one dequant pass serve every gate of a phase.
///
/// The blob keeps the bias as f32 whatever the compute dtype — prepackWeights
/// does not know whether the caller will run f32 or f16 — so a q8 direction
/// converts it once, in the driver.

#include "backend/cpu/rnn/recurrent_internal.h"

namespace nnops::backend::cpu {

namespace {

/// The direction's slice of a per-row scale table, or null for a per-tensor
/// one. `rows` is one direction's row count; a table wider than that is per
/// row, anything narrower is shared by every direction.
inline const float* q8_scale_rows(const QuantParams& q, int64_t d,
                                  int64_t rows) noexcept
{
    if (q.scale_data == nullptr) { return nullptr; }
    return (q.num_scales > rows) ? q.scale_data + d * rows : q.scale_data;
}

/// The same interleave for a quantized operand's per-row scales, so column j of
/// a packed panel is scaled by the scale of the gate row it came from. Null
/// means a per-tensor scalar; the pad lanes take 1, which is inert because the
/// weight bytes beside them are zero.
inline void interleave_scales(const float* src, float scalar, int64_t hidden,
                              int64_t gates, int64_t gate_off, float* dst)
{
    const int64_t q = 8 * gates;
    for (int64_t hb = 0; hb * 8 < hidden; ++hb) {
        for (int64_t g = 0; g < gates; ++g) {
            for (int64_t j = 0; j < 8; ++j) {
                const int64_t h = hb * 8 + j;
                dst[hb * q + g * 8 + j] =
                    (h < hidden && src != nullptr) ? src[(gate_off + g) * hidden + h]
                                                   : scalar;
            }
        }
    }
}

/// Pack one k-block slab set: `nc` columns of a [nc, K] byte matrix, one
/// int8_panel_stride slab per k block.
inline void q8_pack_slabs(int8_t* dst, const int8_t* src, int64_t lds,
                          int64_t k, int64_t cols, bool a_biased)
{
    int64_t off = 0;
    for (int64_t at = 0; at < k; at += kQ8Kc) {
        const int64_t kc = std::min<int64_t>(kQ8Kc, k - at);
        tile_pack_rhs_i8(/*trans=*/true, static_cast<int>(cols),
                         static_cast<int>(kc), dst + off,
                         static_cast<int>(int8_panel_stride(kc)),
                         src + at, static_cast<int>(lds), a_biased);
        off += q8_slab_bytes(cols, kc);
    }
}

}  // namespace

void prepare_q8_direction(const RecurrentParams& p, const RecurrentPackedHeader& h,
                          int64_t d, const TensorView& w, const TensorView& r,
                          const TensorView* bias, int8_t* raw, float* bias_f32,
                          const Q8Regions& dst)
{
    const int64_t G = h.gates;
    const int64_t H = h.hidden;
    const int64_t I = h.input;
    const bool lbr = h.lbr != 0;
    const bool a_biased = int8_a_pack_bias(DataType::s8) != 0;

    const QuantParams& qw = w.quant_params();
    const QuantParams& qr = r.quant_params();
    const float* sw_rows = q8_scale_rows(qw, d, G * H);
    const float* sr_rows = q8_scale_rows(qr, d, G * H);

    int64_t col = 0;
    for (int64_t ph = 0; ph < h.nphases; ++ph) {
        const PhaseKind kind = phase_kind(p.mode, lbr, ph);
        const int64_t gates = phase_gates(kind);
        const int64_t gate_off = phase_gate_offset(p.mode, lbr, ph);
        interleave_gates<int8_t>(w.ptr<int8_t>(static_cast<size_t>(d * G * H)),
                                 w.row_stride_elems(), H, I, gates, gate_off,
                                 raw + col * I);
        interleave_scales(sw_rows, qw.scale, H, gates, gate_off, dst.sw + col);
        col += h.phase_cols[ph];
    }
    q8_pack_slabs(dst.w, raw, I, I, h.total_cols, a_biased);

    for (int64_t ph = 0; ph < h.nphases; ++ph) {
        const PhaseKind kind = phase_kind(p.mode, lbr, ph);
        const int64_t gates = phase_gates(kind);
        const int64_t gate_off = phase_gate_offset(p.mode, lbr, ph);
        interleave_gates<int8_t>(r.ptr<int8_t>(static_cast<size_t>(d * G * H)),
                                 r.row_stride_elems(), H, H, gates, gate_off, raw);
        interleave_scales(sr_rows, qr.scale, H, gates, gate_off, dst.sr[ph]);
        q8_pack_slabs(dst.r[ph], raw, H, H, h.phase_cols[ph], a_biased);
    }

    // The bias is the caller's fp type; the regions are f32.
    const DataType bdt = bias ? bias->data_type() : DataType::f32;
    for (int64_t gi = 0; gi < 2 * G * H; ++gi) {
        float v = 0.0f;
        if (bias != nullptr) {
            v = (bdt == DataType::f16)
                    ? s_load(bias->ptr<half>(static_cast<size_t>(d)) + gi)
                    : bias->ptr<float>(static_cast<size_t>(d))[gi];
        }
        bias_f32[gi] = v;
    }
    const float* wb = bias ? bias_f32 : nullptr;
    const float* rb = bias ? bias_f32 + G * H : nullptr;

    for (int64_t ph = 0; ph < h.nphases; ++ph) {
        const PhaseKind kind = phase_kind(p.mode, lbr, ph);
        const int64_t gates = phase_gates(kind);
        const int64_t gate_off = phase_gate_offset(p.mode, lbr, ph);
        // Rbh belongs after the recurrent product under linear_before_reset, so
        // the h gate's bias vector takes Wb alone here.
        const int64_t h_gate = (kind == PhaseKind::GruZr) ? -1
                             : (kind == PhaseKind::GruAll) ? 2 : 0;
        const int64_t all_gates = (int64_t{1} << gates) - 1;
        const int64_t h_bit = (h_gate < 0) ? 0 : (int64_t{1} << h_gate);
        build_bias<float>(wb, rb, H, gates, lbr ? (all_gates & ~h_bit) : all_gates,
                          gate_off, dst.bias[ph]);
        if (lbr) {
            build_bias<float>(static_cast<const float*>(nullptr), rb, H, gates,
                              h_bit, gate_off, dst.rh);
        }
    }
}

namespace {

/// The per-row dynamic scale of a floating operand: |max| / 127, or 1 for a row
/// that is all zero (its quantized values are zero, so the scale is only a
/// divisor and never observed).
template <typename T>
void quant_row_scales(const T* a, int64_t lds, int64_t rows, int64_t k, float* s)
{
    constexpr float kInv127 = 1.0f / 127.0f;
    for (int64_t m = 0; m < rows; ++m) {
        const T* row = a + m * lds;
        v_f32x8 vmax = v_zero_f32x8();
        int64_t j = 0;
        for (; j + 8 <= k; j += 8) { vmax = v_max(vmax, v_abs(vload8(row + j))); }
        alignas(32) float tmp[8];
        vstore8(tmp, vmax);
        float mx = 0.0f;
        for (int c = 0; c < 8; ++c) { mx = std::max(mx, tmp[c]); }
        for (; j < k; ++j) { mx = std::max(mx, std::fabs(s_load(row + j))); }
        s[m] = mx > 0.0f ? mx * kInv127 : 1.0f;
    }
}

/// The recurrent product on the int8 MMA. A — the hidden state, in `dtype` — is
/// quantized per row from its own max, B is the packed s8 R panels, and the
/// int32 accumulator is dequantized into the fp destination the epilogue reads.
/// The scale is one per row over the whole K, so the dequant is one pass however
/// many k blocks the product took.
template <typename T, PhaseKind Kind>
void run_phase_q8_kernel(const StepCtx<T>& s, const ComputeContext& ctx)
{
    const int64_t tasks = s.m_blocks * s.n_blocks;
    const bool fresh = (s.rec != nullptr);
    const bool a_biased = int8_a_pack_bias(DataType::s8) != 0;

    auto body = [&](int64_t task) {
        const int64_t slot = (s.qs.slots < tasks)
                                 ? static_cast<int64_t>(ctx.cpu.current_thread_id())
                                 : task;
        int8_t* pa = s.qs.a + slot * s.qs.a_bytes;
        int32_t* pc = s.qs.c + slot * s.qs.c_elems;
        float* psa = s.qs.s + slot * s.qs.s_elems;

        const int64_t mblk = task / s.n_blocks;
        const int64_t nblk = task % s.n_blocks;
        const int64_t m0 = mblk * s.mc;
        const int64_t mcnt = std::min<int64_t>(s.mc, s.batch - m0);
        const int64_t n0 = nblk * s.nc;
        const int64_t ncnt = std::min<int64_t>(s.nc, s.cols - n0);
        if (mcnt <= 0 || ncnt <= 0) { return; }

        const T* a = s.a + m0 * s.hp;
        quant_row_scales(a, s.hp, mcnt, s.hidden, psa);

        int64_t slab = 0;
        for (int64_t k = 0, kb = 0; k < s.hidden; k += s.kc, ++kb) {
            const int64_t kc = std::min<int64_t>(s.kc, s.hidden - k);
            tile_pack_quant_lhs_i8(static_cast<int>(mcnt), static_cast<int>(kc),
                                   pa, static_cast<int>(geom_ldd_i8_a(kc)), a + k,
                                   static_cast<int>(s.hp), psa, nullptr, 1.0f, 0,
                                   /*u8=*/false, a_biased);
            const int8_t* pb = s.q_r + slab +
                               (n0 / NR_MAX_I8) * int8_panel_stride(kc);
            tile_mma_pack_i8(static_cast<int>(mcnt), static_cast<int>(ncnt),
                             static_cast<int>(kc), pc, static_cast<int>(s.nc),
                             pa, pb, /*ldb=*/0, INT32_MIN, INT32_MAX,
                             /*zero_mode=*/kb == 0);
            slab += q8_slab_bytes(s.cols, kc);
        }

        T* dst = (fresh ? s.rec : s.in_proj) + m0 * s.ldc + n0;
        const float* sr = s.sr + n0;
        for (int64_t m = 0; m < mcnt; ++m) {
            const int32_t* crow = pc + m * s.nc;
            T* drow = dst + m * s.ldc;
            const float sa = psa[m];
            for (int64_t n = 0; n < ncnt; ++n) {
                const float v = static_cast<float>(crow[n]) * sa * sr[n];
                s_store(drow + n, fresh ? v : s_load(drow + n) + v);
            }
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

void run_phase_q8(const StepCtx<float>& s, PhaseKind kind, const ComputeContext& ctx) {
    switch (kind) {
    case PhaseKind::Rnn:    run_phase_q8_kernel<float, PhaseKind::Rnn>(s, ctx); break;
    case PhaseKind::Lstm:   run_phase_q8_kernel<float, PhaseKind::Lstm>(s, ctx); break;
    case PhaseKind::GruAll: run_phase_q8_kernel<float, PhaseKind::GruAll>(s, ctx); break;
    case PhaseKind::GruZr:  run_phase_q8_kernel<float, PhaseKind::GruZr>(s, ctx); break;
    case PhaseKind::GruH:   run_phase_q8_kernel<float, PhaseKind::GruH>(s, ctx); break;
    }
}

void run_phase_q8(const StepCtx<half>& s, PhaseKind kind, const ComputeContext& ctx) {
    switch (kind) {
    case PhaseKind::Rnn:    run_phase_q8_kernel<half, PhaseKind::Rnn>(s, ctx); break;
    case PhaseKind::Lstm:   run_phase_q8_kernel<half, PhaseKind::Lstm>(s, ctx); break;
    case PhaseKind::GruAll: run_phase_q8_kernel<half, PhaseKind::GruAll>(s, ctx); break;
    case PhaseKind::GruZr:  run_phase_q8_kernel<half, PhaseKind::GruZr>(s, ctx); break;
    case PhaseKind::GruH:   run_phase_q8_kernel<half, PhaseKind::GruH>(s, ctx); break;
    }
}

/// The input projection on the int8 MMA. A is the chunk's X rows — floating and
/// dynamically quantized per row, or s8 with the tensor's own per-row table — and
/// B is W's per-k-block slabs; the dequant writes the fp chunk buffer.
template <typename T>
void project_chunk_q8(const TensorView& x, int64_t x_row0, T* c, int64_t ldc,
                      const int8_t* packed_w, const float* sw, int64_t cols,
                      int64_t input, int64_t rows, int64_t mc, int64_t nc,
                      const RowScales& xs, const Q8Scratch& qs,
                      const ComputeContext& ctx)
{
    const int64_t n_blocks = (cols + nc - 1) / nc;
    const int64_t m_blocks = (rows + mc - 1) / mc;
    const int64_t tasks = m_blocks * n_blocks;
    const bool x_s8 = x.data_type() == DataType::s8;
    const int64_t x_ld = x.row_stride_elems();
    const bool a_biased = int8_a_pack_bias(DataType::s8) != 0;

    auto body = [&](int64_t task) {
        const int64_t slot = (qs.slots < tasks)
                                 ? static_cast<int64_t>(ctx.cpu.current_thread_id())
                                 : task;
        int8_t* pa = qs.a + slot * qs.a_bytes;
        int32_t* pc = qs.c + slot * qs.c_elems;
        float* psa = qs.s + slot * qs.s_elems;

        const int64_t m0 = (task / n_blocks) * mc;
        const int64_t n0 = (task % n_blocks) * nc;
        const int64_t mcnt = std::min<int64_t>(mc, rows - m0);
        const int64_t ncnt = std::min<int64_t>(nc, cols - n0);
        if (mcnt <= 0 || ncnt <= 0) { return; }

        if (x_s8) {
            const float* xt = xs.rows ? xs.rows + x_row0 + m0 : nullptr;
            for (int64_t r = 0; r < mcnt; ++r) {
                psa[r] = xt ? xt[r] : xs.scalar;
            }
        } else {
            quant_row_scales(x.ptr<T>(static_cast<size_t>(x_row0 + m0)), x_ld,
                             mcnt, input, psa);
        }
        const int8_t* xa8 = x_s8
            ? x.ptr<int8_t>(static_cast<size_t>(x_row0 + m0)) : nullptr;
        const T* xa = x_s8 ? nullptr
                           : x.ptr<T>(static_cast<size_t>(x_row0 + m0));

        int64_t slab = 0;
        for (int64_t at = 0, kb = 0; at < input; at += kQ8Kc, ++kb) {
            const int64_t kc = std::min<int64_t>(kQ8Kc, input - at);
            if (x_s8) {
                tile_pack_lhs_i8(/*trans=*/false, static_cast<int>(mcnt),
                                 static_cast<int>(kc), pa,
                                 static_cast<int>(geom_ldd_i8_a(kc)), xa8 + at,
                                 static_cast<int>(x_ld), a_biased);
            } else {
                tile_pack_quant_lhs_i8(static_cast<int>(mcnt), static_cast<int>(kc),
                                       pa, static_cast<int>(geom_ldd_i8_a(kc)),
                                       xa + at, static_cast<int>(x_ld), psa, nullptr,
                                       1.0f, 0, /*u8=*/false, a_biased);
            }
            const int8_t* pb = packed_w + slab +
                               (n0 / NR_MAX_I8) * int8_panel_stride(kc);
            tile_mma_pack_i8(static_cast<int>(mcnt), static_cast<int>(ncnt),
                             static_cast<int>(kc), pc, static_cast<int>(nc),
                             pa, pb, /*ldb=*/0, INT32_MIN, INT32_MAX,
                             /*zero_mode=*/kb == 0);
            slab += q8_slab_bytes(cols, kc);
        }

        T* dst = c + m0 * ldc + n0;
        const float* swp = sw + n0;
        for (int64_t m = 0; m < mcnt; ++m) {
            const int32_t* crow = pc + m * nc;
            T* drow = dst + m * ldc;
            const float sa = psa[m];
            for (int64_t n = 0; n < ncnt; ++n) {
                s_store(drow + n, static_cast<float>(crow[n]) * sa * swp[n]);
            }
        }
    };

    if (tasks <= 1) {
        body(0);
    } else {
        ctx.cpu.run(0, tasks, body);
    }
}

template void project_chunk_q8<float>(const TensorView&, int64_t, float*, int64_t,
                                      const int8_t*, const float*, int64_t, int64_t,
                                      int64_t, int64_t, int64_t, const RowScales&,
                                      const Q8Scratch&, const ComputeContext&);
template void project_chunk_q8<half>(const TensorView&, int64_t, half*, int64_t,
                                     const int8_t*, const float*, int64_t, int64_t,
                                     int64_t, int64_t, int64_t, const RowScales&,
                                     const Q8Scratch&, const ComputeContext&);

}  // namespace nnops::backend::cpu
