/// @file recurrent.cpp
/// @brief The CPU recurrent kernel — RNN, GRU and LSTM behind one driver.
///
/// A recurrent layer's time splits three ways, and only the middle one is a GEMM
/// that has to run a step at a time:
///
///   1. The input projection X·W^T. All of its rows exist up front, so it runs
///      as one GEMM per chunk of timesteps (M = chunk·B) and pays the GEMM's
///      arithmetic intensity rather than a vector-matrix product's.
///   2. H_{t-1}·R^T. Serial in t by definition, so this one does run B rows at a
///      time — but into the chunk buffer with the prepacked R panels, and it is
///      the only per-step pass over [B, gates·H].
///   3. The gate arithmetic. O(gates·H) flops per row: trivial, and ruinous as a
///      separate pass. It is fused into the tile epilogue, so a gate value goes
///      from the MMA accumerator to the activation, the cell update, the
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
#include <cstring>
#include <type_traits>
#include <utility>

namespace nnops::backend::cpu {
namespace {


using namespace nnops::simd;

// =========================================================================
//  Element helpers — one f32 arithmetic surface over f32 and f16
// =========================================================================

template <typename T> constexpr DataType dtype_of() noexcept {
    return std::is_same_v<T, float> ? DataType::f32 : DataType::f16;
}

inline v_f32x8 vload8(const float* p) { return v_load(p); }
inline v_f32x8 vload8(const half* p)  { return v_cvt_f16_to_f32(p); }

inline void vstore8(float* p, const v_f32x8& v) { v_store(p, v); }
inline void vstore8(half* p, const v_f32x8& v)  { v_store(p, v_cvt_f32_to_f16(v)); }

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
};

int64_t gcd64(int64_t a, int64_t b) noexcept {
    while (b != 0) { const int64_t t = a % b; a = b; b = t; }
    return a;
}

int64_t lcm64(int64_t a, int64_t b) noexcept {
    return (a / gcd64(a, b)) * b;
}

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
    int64_t elem_size;
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
inline constexpr uint32_t kRecurrentPackedVersion = 1;
/// Both the header and every region start on this, which is also the packing
/// alignment the panels need.
inline constexpr int64_t kPackedAlign = 64;

inline int64_t align_up64(int64_t bytes) noexcept {
    return (bytes + kPackedAlign - 1) / kPackedAlign * kPackedAlign;
}

inline int64_t packed_header_bytes() noexcept {
    return align_up64(static_cast<int64_t>(sizeof(RecurrentPackedHeader)));
}

/// Where each region of one direction starts, in bytes from the direction's base.
struct PackedWeightOffsets {
    int64_t w = 0;
    int64_t r[2] = {0, 0};
    int64_t bias[2] = {0, 0};
    int64_t rh = 0;
    int64_t dir_bytes = 0;
};

/// The region walk, written once so the packer and the consumer agree by
/// construction rather than by two copies of the same arithmetic.
PackedWeightOffsets packed_weight_offsets(const RecurrentPackedHeader& h) noexcept
{
    PackedWeightOffsets o;
    int64_t off = 0;
    const auto take = [&](int64_t elems) {
        const int64_t at = off;
        off = align_up64(off + elems * h.elem_size);
        return at;
    };
    // A panel region is np·ldd, which is k rounded up to nr — not k, so the
    // geometry call, not the raw extent, is what sizes it.
    const DataType dt = h.elem_size == 4 ? DataType::f32 : DataType::f16;
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
    h.elem_size = static_cast<int64_t>(data_type_size(p.dtype));
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

/// A tile smaller than this does less MMA work than the dispatch around it
/// costs, and a step's grid is dispatched once per phase per timestep — so the
/// floor is what keeps a large team from being slower than a small one on the
/// shapes where B and the phase width together do not amount to much.
constexpr int64_t kMinTileMacs = int64_t{1} << 16;

/// Per-timestep tile grid: split B and the phase width so the padded cell count
/// per busy worker is smallest. A single column block takes the whole width even
/// when that is not a multiple of the grid; every other block starts on it.
void solve_tiles(int64_t rows, int64_t cols, int64_t k, int64_t n_align,
                 int64_t threads, PhaseGeom& g)
{
    threads = std::max<int64_t>(threads, 1);
    g.m_blocks = 1; g.n_blocks = 1; g.mc = rows; g.nc = cols;

    const int64_t max_tasks =
        std::max<int64_t>(1, rows * cols * k / kMinTileMacs);

    int64_t best_pad = -1, best_used = 1;
    const int64_t max_nb = std::max<int64_t>(
        1, std::min<int64_t>(std::min<int64_t>(threads, max_tasks), cols / n_align));

    for (int64_t nb = 1; nb <= max_nb; ++nb) {
        int64_t nc_t = cols;
        if (nb > 1) {
            const int64_t share = (cols + nb - 1) / nb;
            nc_t = std::min<int64_t>(
                ((share + n_align - 1) / n_align) * n_align, cols);
        }
        const int64_t nblk = (nc_t >= cols) ? 1 : (cols + nc_t - 1) / nc_t;

        const int64_t mb = std::max<int64_t>(
            1, std::min<int64_t>(std::min<int64_t>(threads / nblk, rows),
                                 max_tasks / nblk));
        const int64_t mc_t = (rows + mb - 1) / mb;
        const int64_t mblk = (rows + mc_t - 1) / mc_t;

        const int64_t used = std::min<int64_t>(mblk * nblk, threads);
        const int64_t pad = (mblk * mc_t) * (nblk * nc_t);

        if (best_pad < 0 || pad * best_used < best_pad * used) {
            best_pad = pad; best_used = used;
            g.m_blocks = mblk; g.n_blocks = nblk; g.mc = mc_t; g.nc = nc_t;
        }
    }
}

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

/// The recurrent product for one timestep, then the epilogue, over the tile grid.
/// With `rec` set the tile is built from nothing (linear_before_reset's second
/// accumulation); otherwise it adds into the projection.
template <typename T, PhaseKind Kind>
void run_phase(const StepCtx<T>& s, const ComputeContext& ctx)
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
            tile_mma_direct_ldb(static_cast<int>(mcnt), static_cast<int>(ncnt),
                                static_cast<int>(kc),
                                c_tile, static_cast<int>(s.ldc),
                                a + k, static_cast<int>(s.hp),
                                b_panel + k * s.r_nr, s.r_ldd,
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
            tile_mma_direct_ldb(static_cast<int>(mcnt), static_cast<int>(ncnt),
                                static_cast<int>(kcnt),
                                c_tile, static_cast<int>(ldc),
                                a + k, static_cast<int>(x_ld),
                                b_panel + k * w_nr, w_ldd,
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

void run_phase(const StepCtx<float>& s, PhaseKind kind, const ComputeContext& ctx) {
    switch (kind) {
    case PhaseKind::Rnn:    run_phase<float, PhaseKind::Rnn>(s, ctx); break;
    case PhaseKind::Lstm:   run_phase<float, PhaseKind::Lstm>(s, ctx); break;
    case PhaseKind::GruAll: run_phase<float, PhaseKind::GruAll>(s, ctx); break;
    case PhaseKind::GruZr:  run_phase<float, PhaseKind::GruZr>(s, ctx); break;
    case PhaseKind::GruH:   run_phase<float, PhaseKind::GruH>(s, ctx); break;
    }
}

void run_phase(const StepCtx<half>& s, PhaseKind kind, const ComputeContext& ctx) {
    switch (kind) {
    case PhaseKind::Rnn:    run_phase<half, PhaseKind::Rnn>(s, ctx); break;
    case PhaseKind::Lstm:   run_phase<half, PhaseKind::Lstm>(s, ctx); break;
    case PhaseKind::GruAll: run_phase<half, PhaseKind::GruAll>(s, ctx); break;
    case PhaseKind::GruZr:  run_phase<half, PhaseKind::GruZr>(s, ctx); break;
    case PhaseKind::GruH:   run_phase<half, PhaseKind::GruH>(s, ctx); break;
    }
}

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

private:
    char* cur_;
    char* end_;
};

// =========================================================================
//  Driver
// =========================================================================

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

Status build_plan(const RecurrentParams& p, int64_t threads, RecPlan& plan)
{
    const int64_t esz = static_cast<int64_t>(data_type_size(p.dtype));
    const int64_t H = p.hidden;
    const int64_t nr = (esz == 4) ? NR_MAX_F32 : NR_MAX_F16;
    const int64_t kc_cap = (esz == 4) ? KC_CAP_F32 : KC_CAP_F16;

    const bool lbr = p.mode == RecurrentMode::Gru && p.linear_before_reset;
    plan.nphases = num_phases(p.mode, lbr);
    plan.hp = (H + 7) / 8 * 8;
    plan.total_cols = 0;

    for (int64_t ph = 0; ph < plan.nphases; ++ph) {
        PhaseGeom& g = plan.phase[ph];
        g.kind = phase_kind(p.mode, lbr, ph);
        g.gates = phase_gates(g.kind);
        g.q = 8 * g.gates;
        g.cols = ((H + 7) / 8) * g.q;
        g.n_align = lcm64(g.q, nr);
        g.rgeo = matmul_packed_b_geom(g.cols, H, p.dtype);
        if (!g.rgeo.ok()) { return Status::invalid_argument; }
        plan.col_base[ph] = plan.total_cols;
        plan.total_cols += g.cols;
        solve_tiles(p.batch, g.cols, std::min(H, kc_cap), g.n_align, threads, g);
    }

    plan.wgeo = matmul_packed_b_geom(plan.total_cols, p.input, p.dtype);
    if (!plan.wgeo.ok()) { return Status::invalid_argument; }

    plan.kc = std::max<int64_t>(1, std::min<int64_t>(H, kc_cap));

    // Chunk the input projection so its buffer stays inside a shared L3.
    const int64_t budget = int64_t{8} << 20;
    plan.chunk = std::max<int64_t>(
        1, budget / (std::max<int64_t>(p.batch, 1) * plan.total_cols * esz));
    plan.chunk = std::min(plan.chunk, std::max<int64_t>(p.seq, 1));
    return Status::ok;
}

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
    NNOPS_TRY(build_plan(p, ctx.cpu.thread_count(), plan));
    const int64_t hp = plan.hp;
    const int64_t ldc = plan.total_cols;
    const DataType dt = p.dtype;

    // A prepacked W is every weight this driver would build — W and R panels and
    // the bias vectors — so the packing scratch disappears with it.
    const TensorView& w_in = inputs[1];
    const bool prepacked = is_packed_recurrent(w_in.layout());
    const RecurrentPackedHeader* ph_hdr = nullptr;
    if (prepacked) {
        const char* blob = reinterpret_cast<const char*>(w_in.ptr<T>());
        const auto* h = reinterpret_cast<const RecurrentPackedHeader*>(blob);
        if (h->magic != kRecurrentPackedMagic || h->version != kRecurrentPackedVersion ||
            h->elem_size != esz || h->input != I || h->hidden != H ||
            h->directions != D || h->gates != G ||
            h->lbr != static_cast<int64_t>(lbr ? 1 : 0) ||
            w_in.numel() < recurrent_packed_numel(p)) {
            return Status::invalid_argument;
        }
        ph_hdr = h;
    }

    // ---- scratch: one pooled block, carved per direction ------------------
    int64_t elems = 0;
    if (!prepacked) {
        elems += plan.total_cols * std::max(I, H);  // the interleave staging area
        elems += packed_b_numel(plan.total_cols, plan.wgeo.total);
        for (int64_t ph = 0; ph < plan.nphases; ++ph) {
            const PhaseGeom& g = plan.phase[ph];
            elems += packed_b_numel(g.cols, g.rgeo.total);
            elems += g.cols;                        // bias
        }
        if (lbr) { elems += plan.phase[0].cols; }   // Rbh, kept apart
    }
    elems += plan.chunk * B * ldc;              // the input projection
    if (lbr) { elems += B * ldc; }              // h~'s second accumulation
    elems += 2 * B * hp;                        // h_prev, h_next
    if (is_lstm) { elems += 2 * B * hp; }       // c_prev, c_next
    if (is_gru && !lbr) { elems += 2 * B * hp; }  // z, r⊙H

    const size_t bytes = static_cast<size_t>(elems) * sizeof(T) + 4096;
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

        if (prepacked) {
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
            project_chunk<T>(x.ptr<T>(static_cast<size_t>(t0 * B)), x_rows,
                             in_proj, ldc, packed_w,
                             plan.wgeo.ldd, plan.wgeo.nr,
                             nrows * B, plan.total_cols, I, plan.kc,
                             plan.total_cols, ctx);

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
                    run_phase(s, g.kind, ctx);
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

}  // anonymous namespace

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
    if (p.dtype != DataType::f32 && p.dtype != DataType::f16) {
        return Status::unsupported;
    }
    if (p.input <= 0 || p.hidden <= 0 || p.directions <= 0) {
        return Status::invalid_argument;
    }
    return p.dtype == DataType::f32
               ? pack_weights_impl<float>(p, w, r, bias, dst, ctx)
               : pack_weights_impl<half>(p, w, r, bias, dst, ctx);
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
