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

    // ---- quantized route only ----
    bool q8 = false;
    int64_t proj_mc = 0;  ///< the input projection's m-tile
    int64_t proj_nc = 0;  ///< the input projection's n-tile
    int64_t slots = 1;    ///< per-worker scratch slots (see CpuBackend)
    int64_t a_bytes = 0;  ///< packed-A panels per slot, the largest tile's
    int64_t c_elems = 0;  ///< int32 accumulator per slot, the largest tile's
    int64_t s_elems = 0;  ///< A row scales per slot, the largest tile's rows
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
    int64_t elem_size;   ///< 1 for quantized panels, else the fp element size
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
inline constexpr uint32_t kRecurrentPackedVersion = 2;
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

/// The input projection's column tile. int8 has no prepacked B, so the panels
/// are addressed per k-block slab; a ~512-wide column tile keeps one task's
/// int32 accumulator small (and is a whole number of nr-wide panels, which is
/// what lets the kernel address a column block as a panel index — so it is
/// rounded down to one: 512 is 32 panels at nr 16, but not at nr 12).
constexpr int64_t kQ8ProjNc = (512 / NR_MAX_I8) * NR_MAX_I8;

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
    h.q8 = p.weight_dtype == DataType::s8 ? 1 : 0;
    h.elem_size = h.q8 != 0 ? 1 : static_cast<int64_t>(data_type_size(p.dtype));
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

// ---- the quantized weights -------------------------------------------------

/// The direction's slice of a per-row scale table, or null for a per-tensor
/// one. `rows` is one direction's row count; a table wider than that is per
/// row, anything narrower is shared by every direction.
inline const float* q8_scale_rows(const QuantParams& q, int64_t d,
                                  int64_t rows) noexcept
{
    if (q.scale_data == nullptr) { return nullptr; }
    return (q.num_scales > rows) ? q.scale_data + d * rows : q.scale_data;
}

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

/// Interleave, pack and scale one direction's s8 weights. `raw` stages the
/// interleaved byte matrix (total_cols × max(I, H)) and `bias_f32` holds one
/// direction's fp bias in f32 (2·G·H entries) — the blob stores f32 so a pack
/// does not have to know the compute dtype.
inline void prepare_q8_direction(const RecurrentParams& p,
                                 const RecurrentPackedHeader& h, int64_t d,
                                 const TensorView& w, const TensorView& r,
                                 const TensorView* bias, int8_t* raw,
                                 float* bias_f32, const Q8Regions& dst)
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

// ---- the quantized twin ----------------------------------------------------

/// The recurrent product on the int8 MMA. A — the hidden state, in `dtype` — is
/// quantized per row from its own max, B is the packed s8 R panels, and the
/// int32 accumulator is dequantized into the fp destination the epilogue reads.
/// The scale is one per row over the whole K, so the dequant is one pass however
/// many k blocks the product took.
template <typename T, PhaseKind Kind>
void run_phase_q8(const StepCtx<T>& s, const ComputeContext& ctx)
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

template <typename T>
void run_phase_q8(const StepCtx<T>& s, PhaseKind kind, const ComputeContext& ctx) {
    switch (kind) {
    case PhaseKind::Rnn:    run_phase_q8<T, PhaseKind::Rnn>(s, ctx); break;
    case PhaseKind::Lstm:   run_phase_q8<T, PhaseKind::Lstm>(s, ctx); break;
    case PhaseKind::GruAll: run_phase_q8<T, PhaseKind::GruAll>(s, ctx); break;
    case PhaseKind::GruZr:  run_phase_q8<T, PhaseKind::GruZr>(s, ctx); break;
    case PhaseKind::GruH:   run_phase_q8<T, PhaseKind::GruH>(s, ctx); break;
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

Status build_plan(const RecurrentParams& p, const ComputeContext& ctx, RecPlan& plan)
{
    const int64_t threads = std::max<int64_t>(ctx.cpu.thread_count(), 1);
    const int64_t esz = static_cast<int64_t>(data_type_size(p.dtype));
    const int64_t H = p.hidden;
    plan.q8 = p.weight_dtype == DataType::s8;
    const int64_t nr = plan.q8 ? NR_MAX_I8
                               : ((esz == 4) ? NR_MAX_F32 : NR_MAX_F16);
    const int64_t kc_cap = plan.q8 ? kQ8Kc
                                   : ((esz == 4) ? KC_CAP_F32 : KC_CAP_F16);

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
        if (!plan.q8) {
            g.rgeo = matmul_packed_b_geom(g.cols, H, p.dtype);
            if (!g.rgeo.ok()) { return Status::invalid_argument; }
        }
        plan.col_base[ph] = plan.total_cols;
        plan.total_cols += g.cols;
        solve_tiles(p.batch, g.cols, std::min(H, kc_cap), g.n_align, threads, g);
    }

    if (!plan.q8) {
        plan.wgeo = matmul_packed_b_geom(plan.total_cols, p.input, p.dtype);
        if (!plan.wgeo.ok()) { return Status::invalid_argument; }
    }

    plan.kc = std::max<int64_t>(1, std::min<int64_t>(H, kc_cap));

    // Chunk the input projection so its buffer stays inside a shared L3.
    const int64_t budget = int64_t{8} << 20;
    plan.chunk = std::max<int64_t>(
        1, budget / (std::max<int64_t>(p.batch, 1) * plan.total_cols * esz));
    plan.chunk = std::min(plan.chunk, std::max<int64_t>(p.seq, 1));

    if (plan.q8) {
        // The projection's m-tile, by the same rule project_chunk uses: start
        // wide, then narrow until the grid is deep enough for the team.
        const int64_t rows = plan.chunk * std::max<int64_t>(p.batch, 1);
        const int64_t threads_n = std::max<int64_t>(threads, 1);
        int64_t mc = 64;
        for (;;) {
            const int64_t mb = (rows + mc - 1) / mc;
            if (mb >= threads_n || mc <= 8) { break; }
            mc -= 8;
        }
        plan.proj_mc = std::min<int64_t>(mc, std::max<int64_t>(rows, 1));
        // A column tile keeps one task's int32 accumulator bounded; the panels
        // are addressed per k-block slab, so any nr multiple is addressable.
        plan.proj_nc = std::min<int64_t>(plan.total_cols, kQ8ProjNc);

        // Scratch slots, by the library's rule: thread ids bound the slots when
        // the grid hands one worker several items, otherwise every task owns
        // its slice. Sized for the largest tile any phase and the projection
        // can produce, since the phases run one after another.
        const int64_t proj_n_blocks = (plan.total_cols + plan.proj_nc - 1) / plan.proj_nc;
        int64_t tasks_max = (rows + plan.proj_mc - 1) / plan.proj_mc * proj_n_blocks;
        for (int64_t ph = 0; ph < plan.nphases; ++ph) {
            tasks_max = std::max<int64_t>(
                tasks_max, plan.phase[ph].m_blocks * plan.phase[ph].n_blocks);
        }
        const int64_t workers = std::max<int64_t>(ctx.cpu.worker_count(), 1);
        plan.slots = (ctx.cpu.thread_slot_count() > 0 && workers < tasks_max)
                         ? workers : std::max<int64_t>(tasks_max, 1);
        plan.a_bytes = num_panels_max(static_cast<int>(plan.proj_mc), MR_I8, 3) *
                       geom_ldd_i8_a(std::min<int64_t>(kQ8Kc, std::max<int64_t>(p.input, 1)));
        plan.a_bytes = std::max<int64_t>(
            plan.a_bytes, num_panels_max(static_cast<int>(p.batch), MR_I8, 3) *
                          geom_ldd_i8_a(std::min<int64_t>(kQ8Kc, H)));
        plan.c_elems = plan.proj_mc * plan.proj_nc;
        plan.s_elems = plan.proj_mc;
        for (int64_t ph = 0; ph < plan.nphases; ++ph) {
            const PhaseGeom& g = plan.phase[ph];
            plan.c_elems = std::max<int64_t>(plan.c_elems, g.mc * g.nc);
            plan.s_elems = std::max<int64_t>(plan.s_elems, g.mc);
        }
    }
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
    if (p.weight_dtype == DataType::s8) {
        return pack_weights_q8(p, w, r, bias, dst, ctx);
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
