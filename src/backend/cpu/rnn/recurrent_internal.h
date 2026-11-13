#pragma once
/// @file recurrent_internal.h
/// @brief Machinery the recurrent driver, its plan, its packer and its kernels
///        share.
///
/// recurrent.h is the backend interface; this header is the implementation side
/// of it, included only by the recurrent translation units:
///
///   recurrent_plan.cpp  — build_plan: the tile grid, the chunk, the q8 scratch
///   recurrent_pack.cpp  — the packed-weight blob: layout, header, packers
///   recurrent_step.cpp  — one phase of one timestep (fp): MMA + fused epilogue
///   recurrent_q8.cpp    — the int8 twin: weight prep, projection and step
///   recurrent.cpp       — recurrent_cpu and the driver that walks the timeline
///
/// What lives here is what those files must agree on: the gate interleave, the
/// phase decomposition, the packed-blob region walk, the step context each
/// kernel fills, and the entry points one file calls in another.

#include "backend/cpu/rnn/recurrent.h"

#include "backend/cpu/common/common_utils.hpp"
#include "backend/cpu/common/memory_pool.hpp"
#include "backend/cpu/matmul/matmul.h"
#include "backend/cpu/matmul/matmul_helper.h"
#include "backend/cpu/simd_kernel/activation_kernels.hpp"
#include "nnops/detail/assert.hpp"
#include "nnops/detail/simd/simd.hpp"
#include "nnops/detail/validate.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

namespace nnops::backend::cpu {

using namespace nnops::simd;

// =========================================================================
//  Element helpers — one f32 arithmetic surface over f32 and f16
// =========================================================================

template <typename T> constexpr DataType dtype_of() noexcept {
    if constexpr (std::is_same_v<T, float>) { return DataType::f32; }
    else if constexpr (std::is_same_v<T, bfloat16>) { return DataType::bf16; }
    else { return DataType::f16; }
}

inline v_f32x8 vload8(const float* p) { return v_load(p); }
inline v_f32x8 vload8(const half* p)  { return v_cvt_f16_to_f32(p); }
inline v_f32x8 vload8(const bfloat16* p) { return v_load(p); }

inline void vstore8(float* p, const v_f32x8& v) { v_store(p, v); }
inline void vstore8(half* p, const v_f32x8& v)  { v_store(p, v_cvt_f32_to_f16(v)); }
inline void vstore8(bfloat16* p, const v_f32x8& v) { v_store(p, v); }

// =========================================================================
//  Phases — which gates one pass covers
// =========================================================================

/// GruAll is the GRU with linear_before_reset, whose three recurrent products
/// all take H_{t-1} and so are one pass; without it h~'s operand is r⊙H_{t-1},
/// which is only known once r is, so that GRU splits into GruZr then GruH.
enum class PhaseKind : uint8_t { Rnn, Lstm, GruAll, GruZr, GruH };

constexpr int64_t phase_gates(PhaseKind k) noexcept {
    return k == PhaseKind::Rnn    ? 1
         : k == PhaseKind::Lstm   ? 4
         : k == PhaseKind::GruAll ? 3
         : k == PhaseKind::GruZr  ? 2
                                  : 1;
}

constexpr int64_t num_phases(RecurrentMode m, bool lbr) noexcept {
    return (m == RecurrentMode::Gru && !lbr) ? 2 : 1;
}

constexpr PhaseKind phase_kind(RecurrentMode m, bool lbr, int64_t phase) noexcept {
    return m == RecurrentMode::Rnn  ? PhaseKind::Rnn
         : m == RecurrentMode::Lstm ? PhaseKind::Lstm
         : lbr                      ? PhaseKind::GruAll
         : (phase == 0 ? PhaseKind::GruZr : PhaseKind::GruH);
}

/// ONNX gate index of a phase's first row within W, R and B.
constexpr int64_t phase_gate_offset(RecurrentMode m, bool lbr, int64_t phase) noexcept {
    return (m == RecurrentMode::Gru && !lbr && phase == 1) ? 2 : 0;
}

// =========================================================================
//  Plan
// =========================================================================

/// One pass over a range of the gate row: its interleaved width, its tile grid,
/// its weights.
struct PhaseGeom {
    PhaseKind kind = PhaseKind::Rnn;
    int64_t gates = 1;
    int64_t q = 8;        ///< 8·gates — the column stride of a hidden block
    int64_t cols = 0;     ///< q · ⌈H/8⌉
    int64_t n_align = 0;  ///< lcm(q, nr): the column-tile grid
    MatMulPackedB rgeo;   ///< packed R [cols, H]
    int64_t m_blocks = 1, n_blocks = 1, mc = 1, nc = 1;
};

struct RecPlan {
    PhaseGeom phase[2];
    int64_t nphases = 1;
    int64_t col_base[2] = {0, 0};  ///< phase start within the gate row
    int64_t total_cols = 0;        ///< the gate row stride
    int64_t hp = 0;                ///< hidden padded to blocks of 8
    int64_t kc = 0;                ///< recurrent k-block
    int64_t chunk = 0;             ///< timesteps per input-projection chunk
    /// X·W^T as one matrix for the whole gate row, so the projection is one GEMM.
    MatMulPackedB wgeo;

    // ---- quantized route only ----
    bool q8 = false;
    int64_t proj_mc = 0;  ///< the input projection's m-tile
    int64_t proj_nc = 0;  ///< the input projection's n-tile
    int64_t slots = 1;    ///< per-worker scratch slots (see CpuBackend)
    int64_t a_bytes = 0;  ///< packed-A panels per slot, the largest tile's
    int64_t c_elems = 0;  ///< int32 accumulator per slot, the largest tile's
    int64_t s_elems = 0;  ///< A row scales per slot, the largest tile's rows
};

/// ⌈total / n⌉ · n — a packed-B descriptor's numel.
inline int64_t packed_b_numel(int64_t n, int64_t total) noexcept {
    return ((total + n - 1) / n) * n;
}

template <typename T>
TensorView packed_view(int64_t n, int64_t total, T* data) noexcept {
    int64_t shape[2] = {n, packed_b_numel(n, total) / n};
    return TensorView(std::span<const int64_t>(shape, 2), dtype_of<T>(), data,
                      shape[1] * static_cast<int64_t>(sizeof(T)));
}

// =========================================================================
//  Prepacked weights
// =========================================================================

/// The blob prepackWeights() hands back: one header, then a region per
/// direction holding that direction's packed W, packed R, bias vectors and —
/// under linear_before_reset — the h gate's Rbh. The header carries every
/// number a compute() would otherwise derive from W's shape, so a blob packed
/// for another geometry is rejected rather than misread.
struct RecurrentPackedHeader {
    uint32_t magic;
    uint32_t version;
    int64_t elem_size;   ///< 1 for quantized panels, else the fp element size
    int64_t dtype;       ///< the fp element's DataType: f16 and bf16 share a size
    int64_t q8;          ///< W and R are s8: int8 panels and f32 scales
    int64_t input;
    int64_t hidden;
    int64_t directions;
    int64_t gates;
    int64_t lbr;
    int64_t total_cols;
    int64_t nphases;
    int64_t phase_cols[2];
    int64_t dir_bytes;
};

inline constexpr uint32_t kRecurrentPackedMagic = 0x5043524e;  // 'NRCP'
inline constexpr uint32_t kRecurrentPackedVersion = 3;
/// Both the header and every region start on this, which is also the packing
/// alignment the panels need.
inline constexpr int64_t kPackedAlign = 64;

inline int64_t align_up64(int64_t bytes) noexcept {
    return (bytes + kPackedAlign - 1) / kPackedAlign * kPackedAlign;
}

inline int64_t packed_header_bytes() noexcept {
    return align_up64(static_cast<int64_t>(sizeof(RecurrentPackedHeader)));
}

/// The quantized route's k-block. The int8 MMA takes no RHS stride, so a panel
/// set is addressed at int8_panel_stride(kc) for the k block it was packed for;
/// fixing the block size makes the boundaries — and so the packed blob — a
/// function of K alone, independent of the batch, the sequence and the team.
constexpr int64_t kQ8Kc = KC_CAP_I8;

inline int64_t q8_panels(int64_t cols) noexcept {
    return (cols + NR_MAX_I8 - 1) / NR_MAX_I8;
}

/// One k-block's slab: every column panel at this block's panel stride.
inline int64_t q8_slab_bytes(int64_t cols, int64_t kc) noexcept {
    return q8_panels(cols) * int8_panel_stride(kc);
}

/// Every k-block's slab, summed, for a K-wide operand.
inline int64_t q8_packed_bytes(int64_t k, int64_t cols) noexcept {
    int64_t bytes = 0;
    for (int64_t at = 0; at < k; at += kQ8Kc) {
        bytes += q8_slab_bytes(cols, std::min<int64_t>(kQ8Kc, k - at));
    }
    return bytes;
}

/// Where each region of one direction starts, in bytes from the direction's base.
struct PackedWeightOffsets {
    int64_t w = 0;
    int64_t r[2] = {0, 0};
    int64_t bias[2] = {0, 0};
    int64_t rh = 0;
    int64_t sw = 0;          ///< quantized only: interleaved per-column scales
    int64_t sr[2] = {0, 0};  ///< quantized only, one per phase
    int64_t dir_bytes = 0;
};

/// The region walk, written once so the packer and the consumer agree by
/// construction rather than by two copies of the same arithmetic.
PackedWeightOffsets packed_weight_offsets(const RecurrentPackedHeader& h) noexcept;
RecurrentPackedHeader packed_header(const RecurrentParams& p) noexcept;

// =========================================================================
//  Weight preparation
// =========================================================================

/// Copy a phase's `gates` rows starting at ONNX gate `gate_off` out of a
/// gate-major matrix into the interleaved order. `src` is [G·H, K] with row
/// stride `lds`, `dst` is [cols, K] dense. Hidden units past H read as 0, which
/// is what makes the padded columns inert everywhere downstream.
template <typename T>
void interleave_gates(const T* src, int64_t lds, int64_t hidden, int64_t k,
                      int64_t gates, int64_t gate_off, T* dst)
{
    const int64_t q = 8 * gates;
    const size_t row_bytes = static_cast<size_t>(k) * sizeof(T);
    for (int64_t hb = 0; hb * 8 < hidden; ++hb) {
        for (int64_t g = 0; g < gates; ++g) {
            for (int64_t j = 0; j < 8; ++j) {
                const int64_t h = hb * 8 + j;
                T* d = dst + (hb * q + g * 8 + j) * k;
                if (h < hidden) {
                    std::memcpy(d, src + ((gate_off + g) * hidden + h) * lds, row_bytes);
                } else {
                    std::memset(d, 0, row_bytes);
                }
            }
        }
    }
}

/// The gate row's bias in the interleaved order: Wb + Rb, with `rb_mask` saying
/// which gates take Rb (bit g). The GRU h gate drops it under
/// linear_before_reset, where Rbh belongs after the recurrent product. Either
/// operand may be absent, and an all-zero mask leaves the other gates' slots at
/// zero, which is how the Rbh-only vector is built.
template <typename T>
void build_bias(const T* wb, const T* rb, int64_t hidden, int64_t gates,
                int64_t rb_mask, int64_t gate_off, T* dst)
{
    const int64_t q = 8 * gates;
    for (int64_t hb = 0; hb * 8 < hidden; ++hb) {
        for (int64_t g = 0; g < gates; ++g) {
            const bool use_rb = rb != nullptr && ((rb_mask >> g) & 1) != 0;
            for (int64_t j = 0; j < 8; ++j) {
                const int64_t h = hb * 8 + j;
                float v = 0.0f;
                if (h < hidden) {
                    const int64_t gi = (gate_off + g) * hidden + h;
                    if (wb) { v += s_load(wb + gi); }
                    if (use_rb) { v += s_load(rb + gi); }
                }
                s_store(dst + hb * q + g * 8 + j, v);
            }
        }
    }
}

// =========================================================================
//  Quantized weights
// =========================================================================

/// Where a direction's quantized pieces live. The same writer serves the
/// per-call scratch and the blob, so the two layouts cannot drift.
struct Q8Regions {
    int8_t* w = nullptr;
    int8_t* r[2] = {nullptr, nullptr};
    float* sw = nullptr;
    float* sr[2] = {nullptr, nullptr};
    float* bias[2] = {nullptr, nullptr};
    float* rh = nullptr;
};

/// A quantized activation's per-row scales: a caller table when the tensor
/// carries one per row, else its per-tensor scalar.
struct RowScales {
    const float* rows = nullptr;
    float scalar = 1.0f;
};

inline RowScales row_scales_of(const QuantParams& q, int64_t rows) noexcept
{
    RowScales r;
    if (q.scale_data != nullptr && q.num_scales >= rows && rows > 0) {
        r.rows = q.scale_data;
    } else {
        r.scalar = q.scale;
    }
    return r;
}

/// Interleave, pack and scale one direction's s8 weights. `raw` stages the
/// interleaved byte matrix (total_cols × max(I, H)) and `bias_f32` holds one
/// direction's fp bias in f32 (2·G·H entries) — the blob stores f32 so a pack
/// does not have to know the compute dtype.
void prepare_q8_direction(const RecurrentParams& p, const RecurrentPackedHeader& h,
                          int64_t d, const TensorView& w, const TensorView& r,
                          const TensorView* bias, int8_t* raw, float* bias_f32,
                          const Q8Regions& dst);

/// The per-worker scratch the int8 tiles need: the packed-A panels, the int32
/// accumulator and the A row scales, one slot each.
struct Q8Scratch {
    int8_t* a = nullptr;
    int32_t* c = nullptr;
    float* s = nullptr;
    int64_t slots = 1;
    int64_t a_bytes = 0;
    int64_t c_elems = 0;
    int64_t s_elems = 0;
};

// =========================================================================
//  One phase of one timestep
// =========================================================================

/// Everything a phase reads or writes for a single timestep.
template <typename T>
struct StepCtx {
    int64_t batch = 0;
    int64_t hidden = 0;  ///< the recurrent product's K, and Y's row width
    int64_t hp = 0;      ///< hidden padded to blocks of 8: the state row stride
    int64_t cols = 0;    ///< the gate row's interleaved width
    int64_t kc = 0;
    int64_t ldc = 0;     ///< the gate row stride (RecPlan::total_cols)
    int64_t m_blocks = 1, n_blocks = 1, mc = 1, nc = 1;

    const T* a = nullptr;         ///< recurrent input [batch, hp]
    T* in_proj = nullptr;         ///< Xt·W^T + Wb (+ Rb), this timestep
    T* rec = nullptr;             ///< Ht-1·Rh^T under linear_before_reset
    const T* packed_r = nullptr;  ///< R panels, whole K
    int64_t r_ldd = 0;
    int64_t r_nr = 0;
    const T* bias = nullptr;      ///< the gate row's interleaved bias
    const T* rh_bias = nullptr;   ///< GRU h gate, linear_before_reset only

    // ---- quantized route only ----
    const int8_t* q_r = nullptr;  ///< R's int8 slabs, per k block
    const float* sr = nullptr;    ///< R's interleaved per-column scales
    Q8Scratch qs;                 ///< the int8 scratch, one slot per worker

    T* h_prev = nullptr;
    T* h_next = nullptr;
    T* c_prev = nullptr;
    T* c_next = nullptr;

    T* zb = nullptr;   ///< GRU z, kept for the h~ phase's blend
    T* aux = nullptr;  ///< GRU r⊙Ht-1, the h~ phase's recurrent input

    T* y_row0 = nullptr;  ///< Y rows for batch 0 at this (t, d)
    int64_t y_stride = 0;

    const int32_t* seq_len = nullptr;
    int64_t t = 0;
    float clip = 0.0f;
    RnnActivation act_f = RnnActivation::Sigmoid;
    RnnActivation act_g = RnnActivation::Tanh;
    RnnActivation act_h = RnnActivation::Tanh;
    float alpha_f = 0.0f, alpha_g = 0.0f, alpha_h = 0.0f;
    bool input_forget = false;
    bool linear_before_reset = false;
};

// =========================================================================
//  Entry points one translation unit calls in another
// =========================================================================

/// The plan: the tile grid per phase, the chunk, and the q8 scratch geometry.
Status build_plan(const RecurrentParams& p, const ComputeContext& ctx, RecPlan& plan);

/// The fused epilogue over one tile: gate math, state update, Y store.
/// Instantiated for f32/half/bf16 × the five phases in recurrent_step.cpp.
template <typename T, PhaseKind Kind>
void epilogue_tile(const StepCtx<T>& s, int64_t m0, int64_t mcnt,
                   int64_t n0, int64_t ncnt);

/// The recurrent product for one timestep, then the epilogue, over the tile grid.
void run_phase(const StepCtx<float>& s, PhaseKind kind, const ComputeContext& ctx);
void run_phase(const StepCtx<half>& s, PhaseKind kind, const ComputeContext& ctx);
void run_phase(const StepCtx<bfloat16>& s, PhaseKind kind, const ComputeContext& ctx);

/// The int8 twin of run_phase.
void run_phase_q8(const StepCtx<float>& s, PhaseKind kind, const ComputeContext& ctx);
void run_phase_q8(const StepCtx<half>& s, PhaseKind kind, const ComputeContext& ctx);
void run_phase_q8(const StepCtx<bfloat16>& s, PhaseKind kind, const ComputeContext& ctx);

/// The input projection over one chunk on the fp MMA. Instantiated for f32,
/// half and bf16 in recurrent_step.cpp.
template <typename T>
void project_chunk(const T* x, int64_t x_ld, T* c, int64_t ldc, const T* packed_w,
                   int64_t w_ldd, int64_t w_nr, int64_t rows, int64_t cols,
                   int64_t input, int64_t kc, int64_t nc, const ComputeContext& ctx);

/// The input projection over one chunk on the int8 MMA. Instantiated for f32,
/// half and bf16 in recurrent_q8.cpp.
template <typename T>
void project_chunk_q8(const TensorView& x, int64_t x_row0, T* c, int64_t ldc,
                      const int8_t* packed_w, const float* sw, int64_t cols,
                      int64_t input, int64_t rows, int64_t mc, int64_t nc,
                      const RowScales& xs, const Q8Scratch& qs,
                      const ComputeContext& ctx);

}  // namespace nnops::backend::cpu
