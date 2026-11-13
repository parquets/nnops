/// @file recurrent_pack.cpp
/// @brief The packed-weight blob: its layout, its header, and the packers.
///
/// prepackWeights() runs the interleave and pack passes compute() would run and
/// writes them into a caller-owned buffer — packed W, packed R, the interleaved
/// bias vectors and, under linear_before_reset, Rbh — one region set per
/// direction behind a header that carries every geometry number compute() would
/// otherwise derive from W's shape. The region walk lives here so the packer
/// and the consumer agree by construction; recurrent.cpp only reads the
/// offsets back.

#include "backend/cpu/rnn/recurrent_internal.h"

namespace nnops::backend::cpu {

PackedWeightOffsets packed_weight_offsets(const RecurrentPackedHeader& h) noexcept
{
    PackedWeightOffsets o;
    int64_t off = 0;
    // Byte region, 64-aligned. `h.elem_size` scales the floating regions below;
    // the quantized ones are stated in bytes and take() is only alignment there.
    const auto take_bytes = [&](int64_t bytes) {
        const int64_t at = off;
        off = align_up64(off + bytes);
        return at;
    };
    if (h.q8 != 0) {
        // Panels are per k-block slabs; a blob packed once is readable by any
        // plan, because the k blocks are a function of K alone.
        o.w = take_bytes(q8_packed_bytes(h.input, h.total_cols));
        for (int64_t ph = 0; ph < h.nphases; ++ph) {
            o.r[ph] = take_bytes(q8_packed_bytes(h.hidden, h.phase_cols[ph]));
        }
        o.sw = take_bytes(4 * h.total_cols);
        for (int64_t ph = 0; ph < h.nphases; ++ph) {
            o.sr[ph] = take_bytes(4 * h.phase_cols[ph]);
        }
        // The bias stays f32 here whatever the compute dtype: prepackWeights
        // does not know whether the caller will run f32 or f16.
        for (int64_t ph = 0; ph < h.nphases; ++ph) {
            o.bias[ph] = take_bytes(4 * h.phase_cols[ph]);
        }
        if (h.lbr != 0) { o.rh = take_bytes(4 * h.total_cols); }
        o.dir_bytes = off;
        return o;
    }
    const auto take = [&](int64_t elems) {
        const int64_t at = off;
        off = align_up64(off + elems * h.elem_size);
        return at;
    };
    // A panel region is np·ldd, which is k rounded up to nr — not k, so the
    // geometry call, not the raw extent, is what sizes it.
    const DataType dt = static_cast<DataType>(h.dtype);
    const MatMulPackedB wg = matmul_packed_b_geom(h.total_cols, h.input, dt);
    o.w = take(packed_b_numel(h.total_cols, wg.total));
    for (int64_t ph = 0; ph < h.nphases; ++ph) {
        const MatMulPackedB rg = matmul_packed_b_geom(h.phase_cols[ph], h.hidden, dt);
        o.r[ph] = take(packed_b_numel(h.phase_cols[ph], rg.total));
    }
    for (int64_t ph = 0; ph < h.nphases; ++ph) { o.bias[ph] = take(h.phase_cols[ph]); }
    if (h.lbr != 0) { o.rh = take(h.total_cols); }
    o.dir_bytes = off;
    return o;
}

RecurrentPackedHeader packed_header(const RecurrentParams& p) noexcept
{
    RecurrentPackedHeader h{};
    h.magic = kRecurrentPackedMagic;
    h.version = kRecurrentPackedVersion;
    h.q8 = p.weight_dtype == DataType::s8 ? 1 : 0;
    h.elem_size = h.q8 != 0 ? 1 : static_cast<int64_t>(data_type_size(p.dtype));
    h.dtype = static_cast<int64_t>(h.q8 != 0 ? DataType::f32 : p.dtype);
    h.input = p.input;
    h.hidden = p.hidden;
    h.directions = p.directions;
    h.gates = recurrent_gates(p.mode);
    h.lbr = (p.mode == RecurrentMode::Gru && p.linear_before_reset) ? 1 : 0;
    h.nphases = num_phases(p.mode, h.lbr != 0);
    h.total_cols = 0;
    for (int64_t ph = 0; ph < h.nphases; ++ph) {
        const int64_t gates = phase_gates(phase_kind(p.mode, h.lbr != 0, ph));
        h.phase_cols[ph] = ((p.hidden + 7) / 8) * 8 * gates;
        h.total_cols += h.phase_cols[ph];
    }
    h.dir_bytes = packed_weight_offsets(h).dir_bytes;
    return h;
}

namespace {

/// One direction's weights into the blob: the same interleave and pack passes
/// compute() would run, plus the bias vectors.
template <typename T>
void pack_direction(const RecurrentParams& p, const RecurrentPackedHeader& h,
                    const T* w, int64_t w_lds, const T* r, int64_t r_lds,
                    const T* wb, const T* rb, T* raw, char* base,
                    const ComputeContext& ctx)
{
    const PackedWeightOffsets o = packed_weight_offsets(h);
    const int64_t H = h.hidden;
    const int64_t I = h.input;
    const bool lbr = h.lbr != 0;
    const DataType dt = dtype_of<T>();

    // W: every phase's gate rows into one matrix, then one pack. The packed
    // descriptor comes from the same geometry call compute() uses — its `total`
    // is the padded k, and a view built from the unpadded one is short.
    int64_t col = 0;
    for (int64_t ph = 0; ph < h.nphases; ++ph) {
        const PhaseKind kind = phase_kind(p.mode, lbr, ph);
        interleave_gates(w, w_lds, H, I, phase_gates(kind),
                         phase_gate_offset(p.mode, lbr, ph), raw + col * I);
        col += h.phase_cols[ph];
    }
    {
        int64_t sh[2] = {h.total_cols, I};
        TensorView rv(std::span<const int64_t>(sh, 2), dt, raw,
                      I * static_cast<int64_t>(sizeof(T)));
        const MatMulPackedB wg = matmul_packed_b_geom(h.total_cols, I, dt);
        TensorView pv = packed_view(h.total_cols, wg.total,
                                    reinterpret_cast<T*>(base + o.w));
        matmul_packed_b_pack(rv, pv, /*transpose_b=*/true, ctx);
    }

    for (int64_t ph = 0; ph < h.nphases; ++ph) {
        const PhaseKind kind = phase_kind(p.mode, lbr, ph);
        const int64_t gates = phase_gates(kind);
        const int64_t gate_off = phase_gate_offset(p.mode, lbr, ph);

        interleave_gates(r, r_lds, H, H, gates, gate_off, raw);
        {
            int64_t sh[2] = {h.phase_cols[ph], H};
            TensorView rv(std::span<const int64_t>(sh, 2), dt, raw,
                          H * static_cast<int64_t>(sizeof(T)));
            const MatMulPackedB rg = matmul_packed_b_geom(h.phase_cols[ph], H, dt);
            TensorView pv = packed_view(h.phase_cols[ph], rg.total,
                                        reinterpret_cast<T*>(base + o.r[ph]));
            matmul_packed_b_pack(rv, pv, /*transpose_b=*/true, ctx);
        }

        // Rbh belongs after the recurrent product under linear_before_reset, so
        // the h gate's bias vector here takes Wb alone. The h gate is the third
        // of a merged GRU phase and the only one of a h~ phase.
        const int64_t h_gate = (kind == PhaseKind::GruZr) ? -1
                             : (kind == PhaseKind::GruAll) ? 2 : 0;
        const int64_t all_gates = (int64_t{1} << gates) - 1;
        const int64_t h_bit = (h_gate < 0) ? 0 : (int64_t{1} << h_gate);
        build_bias(wb, rb, H, gates, lbr ? (all_gates & ~h_bit) : all_gates,
                   gate_off, reinterpret_cast<T*>(base + o.bias[ph]));
        if (lbr) {
            build_bias(static_cast<const T*>(nullptr), rb, H, gates, h_bit,
                       gate_off, reinterpret_cast<T*>(base + o.rh));
        }
    }
}

template <typename T>
Status pack_weights_impl(const RecurrentParams& p, const TensorView& w,
                         const TensorView& r, const TensorView* bias,
                         TensorView& dst, const ComputeContext& ctx)
{
    const int64_t need = recurrent_packed_numel(p);
    if (dst.is_empty() || dst.data_type() != p.dtype || dst.numel() < need) {
        return Status::invalid_argument;
    }
    const RecurrentPackedHeader h = packed_header(p);
    const int64_t G = h.gates;

    // The interleave staging area, then the pool the pack's scratch lives in.
    const int64_t elems = h.total_cols * std::max(h.input, h.hidden);
    PoolPtr owner(static_cast<size_t>(elems) * sizeof(T) + 4096);
    if (!owner) { return Status::out_of_memory; }
    T* raw = static_cast<T*>(owner.get());

    char* blob = reinterpret_cast<char*>(dst.ptr<T>());
    std::memcpy(blob, &h, sizeof(h));

    for (int64_t d = 0; d < h.directions; ++d) {
        const T* wb = bias ? bias->ptr<T>(static_cast<size_t>(d)) : nullptr;
        const T* rb = wb ? wb + G * h.hidden : nullptr;
        pack_direction<T>(p, h, w.ptr<T>(static_cast<size_t>(d * G * h.hidden)),
                          w.row_stride_elems(),
                          r.ptr<T>(static_cast<size_t>(d * G * h.hidden)),
                          r.row_stride_elems(), wb, rb, raw,
                          blob + packed_header_bytes() + d * h.dir_bytes, ctx);
    }
    return Status::ok;
}

/// The quantized blob: the fp pack's directions, but W and R go in as int8
/// panels with their per-column scales, and the bias stays f32 — the blob is
/// shared by the f32 and f16 compute routes, and the driver converts it.
Status pack_weights_q8(const RecurrentParams& p, const TensorView& w,
                       const TensorView& r, const TensorView* bias,
                       TensorView& dst, const ComputeContext& ctx)
{
    const int64_t need = recurrent_packed_numel(p);
    if (dst.is_empty() || dst.data_type() != DataType::s8 || dst.numel() < need) {
        return Status::invalid_argument;
    }
    if (w.data_type() != DataType::s8 || r.data_type() != DataType::s8) {
        return Status::invalid_argument;
    }
    const RecurrentPackedHeader h = packed_header(p);
    const int64_t G = h.gates;

    // The interleave staging area, then one direction's bias row in f32.
    const int64_t raw_bytes = align_up64(h.total_cols * std::max(h.input, h.hidden));
    PoolPtr owner(static_cast<size_t>(raw_bytes) +
                  static_cast<size_t>(2 * G * h.hidden) * sizeof(float) + 4096);
    if (!owner) { return Status::out_of_memory; }
    int8_t* raw = static_cast<int8_t*>(owner.get());
    float* bias_f32 =
        reinterpret_cast<float*>(static_cast<char*>(owner.get()) + raw_bytes);

    char* blob = reinterpret_cast<char*>(dst.ptr<int8_t>());
    std::memcpy(blob, &h, sizeof(h));

    for (int64_t d = 0; d < h.directions; ++d) {
        char* base = blob + packed_header_bytes() + d * h.dir_bytes;
        const PackedWeightOffsets o = packed_weight_offsets(h);
        Q8Regions reg;
        reg.w = reinterpret_cast<int8_t*>(base + o.w);
        reg.sw = reinterpret_cast<float*>(base + o.sw);
        for (int64_t ph = 0; ph < h.nphases; ++ph) {
            reg.r[ph] = reinterpret_cast<int8_t*>(base + o.r[ph]);
            reg.sr[ph] = reinterpret_cast<float*>(base + o.sr[ph]);
            reg.bias[ph] = reinterpret_cast<float*>(base + o.bias[ph]);
        }
        reg.rh = h.lbr != 0 ? reinterpret_cast<float*>(base + o.rh) : nullptr;
        prepare_q8_direction(p, h, d, w, r, bias, raw, bias_f32, reg);
    }
    return Status::ok;
}

}  // namespace

int64_t recurrent_packed_numel(const RecurrentParams& p)
{
    const RecurrentPackedHeader h = packed_header(p);
    const int64_t esz = h.elem_size;
    return (packed_header_bytes() + h.directions * h.dir_bytes) / esz;
}

Status recurrent_pack_weights(const RecurrentParams& p, const TensorView& w,
                              const TensorView& r, const TensorView* bias,
                              TensorView& dst, const ComputeContext& ctx)
{
    if (p.dtype != DataType::f32 && p.dtype != DataType::f16 &&
        p.dtype != DataType::bf16) {
        return Status::unsupported;
    }
    if (p.input <= 0 || p.hidden <= 0 || p.directions <= 0) {
        return Status::invalid_argument;
    }
    if (p.weight_dtype == DataType::s8) {
        return pack_weights_q8(p, w, r, bias, dst, ctx);
    }
    if (p.dtype == DataType::f32) {
        return pack_weights_impl<float>(p, w, r, bias, dst, ctx);
    }
    if (p.dtype == DataType::f16) {
        return pack_weights_impl<half>(p, w, r, bias, dst, ctx);
    }
#ifdef NNOPS_ARCH_AARCH64
    return pack_weights_impl<bfloat16>(p, w, r, bias, dst, ctx);
#else
    return Status::unsupported;  // bf16 is aarch64-only (validate() refuses it)
#endif
}

}  // namespace nnops::backend::cpu
