/// @file attention.cpp
/// @brief Shared CPU attention core, plus the three operators' kernel entries.
///
/// See attention.h for the routes, the head mapping and the scratch contract.
/// The only per-operator code here is the attribute → AttentionConfig mapping
/// at the bottom: MultiHeadAttention, GroupQueryAttention and
/// MultiLatentAttention all run the kernels above.

#include "attention.h"
#include "backend/cpu/matmul/matmul_helper.h"                  // tile_mma_direct / tile_pack_rhs + panel constants
#include "backend/cpu/simd_kernel/simd_softmax.hpp"     // softmax + FlashAttention row kernels
#include "backend/cpu/common/memory_pool.hpp"           // internal scratch-memory pool
#include "nnops/ops/group_query_attention.hpp"
#include "nnops/ops/multi_head_attention.hpp"
#include "nnops/ops/multi_latent_attention.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace nnops::backend::cpu {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();
constexpr float kPosInf =  std::numeric_limits<float>::infinity();

// ---- dtype-selected tiling constants -------------------------------------
// k-block length per dtype, mirroring matmul_helper.h so plan and kernels agree.

template <class T>
constexpr int64_t kc_for() {
    return std::is_same_v<T, half> ? KC_F16 : KC_F32;
}

/// Size bound for the packed-B scratch's k-run: min(Kc, k). The loops still
/// step by Kc; pass the longer of the two GEMMs' k.
template <class T>
constexpr int64_t kc_pack_bound(int64_t k) {
    constexpr int64_t kc = std::is_same_v<T, half> ? KC_F16 : KC_F32;
    return k < kc ? k : kc;
}

/// Panel count a uniform packed-B tile of any n <= n_max emits.
template <class T>
constexpr int panels_for_pack(int64_t n_max) {
    constexpr int nr_max = nr_max_flt<T>();
    return static_cast<int>((n_max + nr_max - 1) / nr_max);
}

// ---- head-shape extraction (merged [B,S,H*D] vs explicit [B,H,S,D]) ----

struct HeadShape {
    int64_t B, H, Hkv, Sq, Sk, Sv, Dqk, Dv;
    int64_t Sk_past = 0;  ///< rows in the past segment; Sk counts the whole run
    bool merged;
};

/// The optional mask and past slots. Both are positional: a call that passes a
/// past but no mask still fills the mask slot with an empty view.
inline size_t mask_slot(const AttentionConfig& cfg) { return cfg.v_is_k ? 2u : 3u; }
inline size_t past_slot(const AttentionConfig& cfg) { return cfg.v_is_k ? 3u : 4u; }

inline bool has_past_view(const AttentionConfig& cfg, std::span<const TensorView> inputs)
{
    const size_t p = past_slot(cfg);
    return inputs.size() > p && !inputs[p].is_empty();
}

inline HeadShape resolve_head_shape(const AttentionConfig& cfg,
                                    std::span<const TensorView> inputs)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];

    HeadShape s;
    s.B = Q.shape(0);
    s.H = cfg.num_heads;
    s.Hkv = cfg.kv_heads();
    s.merged = (Q.rank() == 3);
    if (s.merged) {
        s.Sq = Q.shape(1);
        s.Sk = K.shape(1);
        s.Sv = V.shape(1);
        s.Dqk = Q.shape(2) / s.H;
    } else {
        s.Sq = Q.shape(2);
        s.Sk = K.shape(2);
        s.Sv = V.shape(2);
        s.Dqk = Q.shape(3);
    }
    // The value head is Dqk wide unless MLA's latent says otherwise.
    s.Dv = cfg.v_is_k ? cfg.v_head_dim : s.Dqk;
    if (has_past_view(cfg, inputs)) {
        s.Sk_past = inputs[past_slot(cfg)].shape(s.merged ? 1 : 2);
    }
    return s;
}

/// Offset of the value head inside a merged row: a latent K row is Dqk wide
/// per head even though only Dv of those columns are values.
inline int64_t merged_v_head_stride(const AttentionConfig& cfg, int64_t Dqk, int64_t Dv)
{
    return cfg.v_is_k ? Dqk : Dv;
}

/// The optional mask is present only when its view carries data; callers may
/// pass an empty view to mean "absent".
inline bool has_mask_input(const AttentionConfig& cfg, std::span<const TensorView> inputs)
{
    const size_t m = mask_slot(cfg);
    return inputs.size() > m && !inputs[m].is_empty();
}

// ---- scratch slots -------------------------------------------------------
//
// A slot holds one task's scratch. How many are needed depends on the backend:
// one per worker when thread ids are reported (a worker reuses its slot), one
// for a sequential run (only one task is live), and otherwise one per task.
// The kernel's slot index and the plan's slot count are the same decision, so
// both go through these — sizing per task on a sequential run over-allocates
// the scratch by the task count, which at 20+ tasks and a large Br×Bc tile
// pushes it past the pool's cap and turns every call into a fresh malloc.

inline bool scratch_per_worker(const CpuBackend& cpu, int64_t tasks)
{
    const int slots = cpu.thread_slot_count();
    return slots > 0 && slots < tasks;
}

inline int64_t scratch_slot_count(const CpuBackend& cpu, int64_t tasks)
{
    if (scratch_per_worker(cpu, tasks)) { return cpu.thread_slot_count(); }
    return cpu.worker_count() <= 1 ? 1 : tasks;
}

inline int64_t scratch_slot_of(const CpuBackend& cpu, int64_t tasks, int64_t idx)
{
    if (scratch_per_worker(cpu, tasks)) { return cpu.current_thread_id(); }
    return cpu.worker_count() <= 1 ? 0 : idx;
}

// ---- causal row limit ----------------------------------------------------

/// Exclusive end of query row @p i's valid key range under the config: Sk when
/// not causal, else q0 + i + 1 with q0 the row-0 position — cfg.position_offset
/// when given, bottom-right (Sk - Sq) otherwise. @p Sk is the whole run.
inline int64_t causal_limit(const AttentionConfig& cfg, int64_t i,
                            int64_t Sq, int64_t Sk)
{
    if (!cfg.causal) { return Sk; }
    const int64_t q0 = cfg.position_offset >= 0 ? cfg.position_offset : (Sk - Sq);
    const int64_t lim = q0 + i + 1;
    if (lim < 0) { return 0; }
    return lim > Sk ? Sk : lim;
}

template <class T>
inline void row_fill(T* row, int64_t n, float val)
{
    for (int64_t i = 0; i < n; ++i) { simd::s_store(&row[i], val); }
}

// ---- per-(batch,head) tiled kernel ---------------------------------------

template <class T>
void attention_impl(const AttentionConfig& cfg,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    const AttentionPlan& plan,
                    void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const bool has_mask = has_mask_input(cfg, inputs);

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    NNOPS_ASSERT(hs.Sk_past == 0);   // the plan sends a split run to flash
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Sk = hs.Sk, Dqk = hs.Dqk, Dv = hs.Dv;
    const bool merged = hs.merged;

    const float scale = (cfg.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* mask_ptr = has_mask ? inputs[mask_slot(cfg)].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t o_rs = output.row_stride_elems();
    const int64_t v_head_stride = merged_v_head_stride(cfg, Dqk, Dv);

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int64_t kc = kc_for<T>();

    const int mc1 = plan.mc1;
    const int nc1 = plan.nc1;
    const int mc2 = plan.mc2;
    const int nc2 = plan.nc2;

    // Scratch bounds: longest k-block and widest n-block either GEMM sees.
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(
            kc_pack_bound<T>(std::max(Dqk, Sk))) * static_cast<int>(sizeof(T)))
        / static_cast<int>(sizeof(T)));
    const int np_full = std::max(panels_for_pack<T>(nc1),
                                 panels_for_pack<T>(std::min<int64_t>(Dv, nc2)));

    const int64_t scores_elems = Sq * Sk;
    const int64_t pack_elems   = static_cast<int64_t>(np_full) * ldd_b_full;
    const int64_t per_head     = scores_elems + pack_elems;

    const int64_t NG = B * H;
    T* ws_base = static_cast<T*>(workspace);

    const auto run = [&](int64_t idx) {
        const int64_t b = idx / H;
        const int64_t h = idx % H;
        const int64_t hkv = h / (H / Hkv);

        const T* q_head;
        const T* k_head;
        const T* v_head;
        T* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * Dqk;
            k_head = k_ptr + b * Sk * k_rs + hkv * Dqk;
            v_head = v_ptr + b * hs.Sv * v_rs + hkv * v_head_stride;
            o_head = out_ptr + b * Sq * o_rs + h * Dv;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * Hkv + hkv) * Sk * k_rs;
            v_head = v_ptr + (b * Hkv + hkv) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }

        const int64_t slot = scratch_slot_of(ctx.cpu, NG, idx);
        T* scores = ws_base + slot * per_head;
        T* pack_b = scores + scores_elems;

        // ---- GEMM1: scores = Q @ K^T * scale --------------------------
        // k0 == 0 runs zero_mode (overwrite) — no pre-zeroing pass.
        for (int64_t k0 = 0; k0 < Dqk; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Dqk - k0));
            const int ldd_b = static_cast<int>(
                mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                / static_cast<int>(sizeof(T)));
            for (int64_t n = 0; n < Sk; n += nc1) {
                const int actual_nc = static_cast<int>(std::min<int64_t>(nc1, Sk - n));
                tile_pack_rhs(true, actual_nc, actual_kc,
                                      pack_b, ldd_b,
                                      k_head + n * k_rs + k0, static_cast<int>(k_rs), scale);
                for (int64_t m = 0; m < Sq; m += mc1) {
                    const int actual_mc = static_cast<int>(std::min<int64_t>(mc1, Sq - m));
                    tile_mma_direct(actual_mc, actual_nc, actual_kc,
                                            scores + m * Sk + n, static_cast<int>(Sk),
                                            q_head + m * q_rs + k0, static_cast<int>(q_rs),
                                            pack_b, /*ldb=*/-1, /*stage=*/nullptr,
                                            kNegInf, kPosInf, k0 == 0);
                }
            }
        }

        // ---- mask + softmax over the last dim, per row, in-place ------
        // The mask folds into the softmax 3-pass (max → exp+sum → normalize),
        // so scores is never rewritten before normalization — one full-buffer
        // pass cheaper than an explicit `scores += mask`. A causal row only
        // softmaxes its valid prefix; the masked tail is zeroed so GEMM2 reads
        // nothing from it.
        for (int64_t i = 0; i < Sq; ++i) {
            T* row = scores + i * Sk;
            const int64_t n = causal_limit(cfg, i, Sq, Sk);
            if (n == 0) {
                row_fill(row, Sk, 0.0f);
                continue;
            }
            if (mask_ptr) {
                kernel::mask_softmax_process_standard_row<T>(
                    row, mask_ptr + i * Sk, row, n, /*log_softmax=*/false, 1.0f);
            } else {
                kernel::softmax_process_standard_row<T>(
                    row, row, n, /*log_softmax=*/false, 1.0f);
            }
            if (n < Sk) { row_fill(row + n, Sk - n, 0.0f); }
        }

        // ---- GEMM2: output = attn @ V ---------------------------------
        for (int64_t k0 = 0; k0 < Sk; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Sk - k0));
            const int ldd_b = static_cast<int>(
                mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                / static_cast<int>(sizeof(T)));
            for (int64_t n = 0; n < Dv; n += nc2) {
                const int actual_nc = static_cast<int>(std::min<int64_t>(nc2, Dv - n));
                // V is the B side: packed once per (k-block, n-block), then
                // every m-block reuses it.
                tile_pack_rhs(false, actual_nc, actual_kc,
                                      pack_b, ldd_b,
                                      v_head + k0 * v_rs + n, static_cast<int>(v_rs),
                                      1.0f);
                for (int64_t m = 0; m < Sq; m += mc2) {
                    const int actual_mc = static_cast<int>(std::min<int64_t>(mc2, Sq - m));
                    tile_mma_direct(actual_mc, actual_nc, actual_kc,
                                            o_head + m * o_rs + n, static_cast<int>(o_rs),
                                            scores + m * Sk + k0, static_cast<int>(Sk),
                                            pack_b, /*ldb=*/-1, /*stage=*/nullptr, kNegInf, kPosInf, k0 == 0);
                }
            }
        }
    };

    ctx.cpu.run(0, NG, run);
}

// ---- FlashAttention routing + tiling -------------------------------------
//
// The flash path is chosen once the per-head scores matrix is large enough
// that materializing it — plus the fused mask/softmax passes — would thrash
// cache. Br / Bc follow the MLAS L2 heuristic: Bc = L2 / (sizeof(T)·4·(Dqk+Dv))
// and Br = min(Bc, Dqk+Dv), then both capped to Sk / Sq.

constexpr int64_t kFlashMinScores = 1 << 16;  // Sq*Sk threshold (≈ 256 KiB f32)

// Tile caps measured across 1T..20T:
//  - Bc tops out at 256. The L2 formula asks for 512 at D=128 f32, but the
//    S tile (Br·Bc), the accumulator and the packed K no longer share the L2
//    with the K/V stream, and 20T lands 12% below the 256-wide tile.
//  - Causal runs want small tiles, because the work a query block wastes above
//    the diagonal is Br/Sq of the total — but only while the key run is
//    comparable to the query run: the fraction is (Sq-Br)/(2·Sk), so a chunk
//    against a long cache gains nothing and keeps the wide tiles. At least
//    four query blocks are needed for the saved triangle to pay for the tiling
//    overhead (a 128-row MLA block lost 5% at Br 64, its 1024-row prefill
//    gained 7%).
constexpr int kFlashBcMax = 256;
constexpr int kCausalBrMax = 64;
constexpr int kCausalBcMax = 128;

inline bool should_use_flash_attention(int64_t Sq, int64_t Sk)
{
    return Sq * Sk >= kFlashMinScores;
}

inline void resolve_flash_tile_sizes(size_t elem_size, int64_t Sq, int64_t Sk,
                                     int64_t Dqk, int64_t Dv, bool causal,
                                     int& Br, int& Bc)
{
    const size_t l2 = simd::CpuFeatures::get().l2_cache_size();
    const size_t l2_eff = l2 > 0 ? l2 : (1024u * 1024u);  // 1 MiB fallback
    const int64_t head_pairs = Dqk + Dv;  // qk_head_size + v_head_size
    Bc = static_cast<int>(std::min<int64_t>(l2_eff / (elem_size * 4 * static_cast<size_t>(head_pairs)),
                                            kFlashBcMax));
    if (Bc < 1) { Bc = 1; }
    Br = static_cast<int>(std::min<int64_t>(Bc, head_pairs));
    if (causal && Sq >= 4 * kCausalBrMax && Sk <= 2 * Sq) {
        Br = std::min(Br, kCausalBrMax);
        Bc = std::min(Bc, kCausalBcMax);
    }
    Bc = static_cast<int>(std::min<int64_t>(Bc, Sk));
    Br = static_cast<int>(std::min<int64_t>(Br, Sq));
    if (Br < 1) { Br = 1; }
    if (Bc < 1) { Bc = 1; }
}

// Per-task scratch for one Br-row query block, matching the layout inside
// attention_flash_impl: [m:Br][l:Br][s:Br*Bc][o_acc:Br*Dv][pack_b]. The
// packed-B slot serves GEMM1 (K = Dqk, N <= Bc) and GEMM2 (K <= Bc, N = Dv).
template <class T>
inline int64_t flash_per_task_elems(int Br, int Bc, int64_t Dqk, int64_t Dv)
{
    constexpr int nr_max = nr_max_flt<T>();
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(
            kc_pack_bound<T>(std::max<int64_t>(Dqk, Bc))) * static_cast<int>(sizeof(T)))
        / static_cast<int>(sizeof(T)));
    const int np_full = std::max(panels_for_pack<T>(Bc), panels_for_pack<T>(Dv));
    const int pack_elems = np_full * ldd_b_full;
    return static_cast<int64_t>(Br) * 2
         + static_cast<int64_t>(Br) * Bc
         + static_cast<int64_t>(Br) * Dv
         + pack_elems;
}

// ---- tiled FlashAttention kernel -----------------------------------------
//
// One task = one query block (Br rows) of one (batch, head): it streams KV in
// Bc-sized blocks, carrying a running max/sum and an output accumulator, so
// the Sq×Sk scores matrix never exists. Reuses the packed-B route above.

template <class T>
void attention_flash_impl(const AttentionConfig& cfg,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const AttentionPlan& plan,
                          void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const bool has_mask = has_mask_input(cfg, inputs);
    const bool split = has_past_view(cfg, inputs);
    const auto& Kp = split ? inputs[past_slot(cfg)] : K;
    const auto& Vp = split ? inputs[past_slot(cfg) + (cfg.v_is_k ? 0 : 1)] : V;

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Dqk = hs.Dqk, Dv = hs.Dv;
    const int64_t Sk_new = hs.Sk, Sk_past = hs.Sk_past;
    const int64_t Sk = Sk_new + Sk_past;   // the whole [past | new] run
    const bool merged = hs.merged;

    const float scale = (cfg.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* kp_ptr = split ? Kp.ptr<T>() : nullptr;
    const T* vp_ptr = split ? Vp.ptr<T>() : nullptr;
    const T* mask_ptr = has_mask ? inputs[mask_slot(cfg)].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t kp_rs = split ? Kp.row_stride_elems() : k_rs;
    const int64_t vp_rs = split ? Vp.row_stride_elems() : v_rs;
    const int64_t o_rs = output.row_stride_elems();
    const int64_t v_head_stride = merged_v_head_stride(cfg, Dqk, Dv);

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int64_t kc = kc_for<T>();
    constexpr int L = simd::simd_lane_for<T>;

    const int Br = plan.Br;
    const int Bc = plan.Bc;
    const int64_t per_task = flash_per_task_elems<T>(Br, Bc, Dqk, Dv);

    const int64_t NQ = (Sq + Br - 1) / Br;  // number of Br-row query blocks
    const int64_t NG = B * H * NQ;
    T* ws_base = static_cast<T*>(workspace);

    const auto run = [&](int64_t idx) {
        const int64_t qi = idx % NQ;
        const int64_t b  = (idx / NQ) / H;
        const int64_t h  = (idx / NQ) % H;
        const int64_t hkv = h / (H / Hkv);
        const int64_t i0 = qi * Br;
        const int actual_br = static_cast<int>(std::min<int64_t>(Br, Sq - i0));

        const T* q_head;
        const T* k_head;
        const T* v_head;
        T* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * Dqk;
            k_head = k_ptr + b * Sk_new * k_rs + hkv * Dqk;
            v_head = v_ptr + b * hs.Sv * v_rs + hkv * v_head_stride;
            o_head = out_ptr + b * Sq * o_rs + h * Dv;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * Hkv + hkv) * Sk_new * k_rs;
            v_head = v_ptr + (b * Hkv + hkv) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }
        // The run is two contiguous segments, past first; without a past the
        // first is empty. Each is streamed in place, so the caller never has to
        // concatenate.
        const T* k_base[2] = {k_head, k_head};
        const T* v_base[2] = {v_head, v_head};
        int64_t k_str[2] = {k_rs, k_rs};
        int64_t v_str[2] = {v_rs, v_rs};
        int64_t seg_lo[2] = {0, 0};
        int64_t seg_hi[2] = {0, Sk};
        if (split) {
            k_base[0] = merged ? kp_ptr + b * Sk_past * kp_rs + hkv * Dqk
                               : kp_ptr + (b * Hkv + hkv) * Sk_past * kp_rs;
            v_base[0] = merged ? vp_ptr + b * Sk_past * vp_rs + hkv * v_head_stride
                               : vp_ptr + (b * Hkv + hkv) * Sk_past * vp_rs;
            k_str[0] = kp_rs;
            v_str[0] = vp_rs;
            seg_lo[1] = Sk_past;
            seg_hi[0] = Sk_past;
        }

        const int64_t slot = scratch_slot_of(ctx.cpu, NG, idx);
        T* m      = ws_base + slot * per_task;
        T* l      = m + Br;
        T* s      = l + Br;
        T* o_acc  = s + static_cast<int64_t>(Br) * Bc;
        T* pack_b = o_acc + static_cast<int64_t>(Br) * Dv;

        // Reset the running online-softmax state through the scalar simd
        // bridge: std::numeric_limits<T> is not specialized for half (it
        // silently yields 0, not -inf), and static_cast<half>(float) truncates
        // to an integer and stores it as bits on the x86 half wrapper.
        for (int i = 0; i < actual_br; ++i) {
            simd::s_store(&m[i], -std::numeric_limits<float>::infinity());
            simd::s_store(&l[i], 0.0f);
        }
        // Causal early exit: the causal limit grows with the query row, so the
        // last row of the block bounds every row in it. A KV block starting at
        // or past that bound is fully masked for all of them and contributes
        // nothing — for a causal prefill that is the whole upper triangle, and
        // skipping it halves the QK and AV work.
        const int64_t kv_end = cfg.causal
            ? causal_limit(cfg, i0 + actual_br - 1, Sq, Sk)
            : Sk;

        // o_acc is written by the first KV block (zero_mode) — no pre-zeroing
        // pass. The exp_diff rescale may touch it first on the first block;
        // that garbage is fully overwritten by the first MMA.
        bool first_tile = true;
        int64_t j0 = 0;
        while (j0 < kv_end) {
            const int seg = (j0 < seg_lo[1]) ? 0 : 1;
            const int64_t seg_end = std::min(seg_hi[seg], kv_end);
            // The diagonal block is the last one and is rarely a full Bc wide:
            // clamping it bounds the tile to the block's causal rectangle
            // instead of a full-width strip that is mostly masked. A segment
            // boundary also ends a tile, so one tile never spans both.
            const int actual_bc = static_cast<int>(std::min<int64_t>(Bc, seg_end - j0));
            const int64_t koff = j0 - seg_lo[seg];
            const T* kb = k_base[seg];
            const T* vb = v_base[seg];
            const int64_t k_step = k_str[seg];
            const int64_t v_step = v_str[seg];

            // ---- S = Q[i0:i0+Br] @ K[j0:j0+Bc]^T * scale ----------
            // k0 == 0 overwrites (zero_mode) — each KV block owns its S tile.
            for (int64_t k0 = 0; k0 < Dqk; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Dqk - k0));
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                    / static_cast<int>(sizeof(T)));
                tile_pack_rhs(true, actual_bc, actual_kc,
                                      pack_b, ldd_b,
                                      kb + koff * k_step + k0, static_cast<int>(k_step), scale);
                tile_mma_direct(actual_br, actual_bc, actual_kc,
                                        s, actual_bc,
                                        q_head + i0 * q_rs + k0, static_cast<int>(q_rs),
                                        pack_b, /*ldb=*/-1, /*stage=*/nullptr, kNegInf, kPosInf, k0 == 0);
            }

            // ---- additive mask (flat [Sq, Sk]) and/or the causal tail ----
            // Both are per-row, so one pass does both: add the mask over the
            // row's valid prefix, then fill the out-of-range tail with -inf.
            if (mask_ptr || cfg.causal) {
                for (int i = 0; i < actual_br; ++i) {
                    const int64_t lim = causal_limit(cfg, i0 + i, Sq, Sk);
                    int64_t n = lim - j0;
                    if (n < 0) { n = 0; }
                    if (n > actual_bc) { n = actual_bc; }
                    T* srow = s + i * actual_bc;
                    if (mask_ptr) {
                        const T* mrow = mask_ptr + (i0 + i) * Sk + j0;
                        int64_t j = 0;
                        for (; j + L <= n; j += L) {
                            simd::v_store(srow + j, simd::v_add(simd::v_load(srow + j), simd::v_load(mrow + j)));
                        }
                        for (; j < n; ++j) {
                            simd::s_store(&srow[j], simd::s_load(&srow[j]) + simd::s_load(&mrow[j]));
                        }
                    }
                    row_fill(srow + n, actual_bc - n, -std::numeric_limits<float>::infinity());
                }
            }

            // ---- online-softmax update + accumulate into o_acc -----
            for (int i = 0; i < actual_br; ++i) {
                T* srow = s + i * actual_bc;
                const float rowmax = kernel::softmax_row_max<T>(srow, actual_bc);
                if (rowmax == kNegInf) {
                    // The whole tile is out of range for this row (a causal
                    // row above the diagonal): S must be zero or the AV GEMM
                    // would add -inf·v.
                    row_fill(srow, actual_bc, 0.0f);
                    continue;
                }
                const float old_m = simd::s_load(&m[i]);
                const float new_m = std::max(old_m, rowmax);
                const float exp_diff = std::exp(old_m - new_m);  // 0 on the first KV block
                simd::s_store(&m[i], new_m);
                const float rowsum = kernel::softmax_row_exp_sum<T>(srow, srow, actual_bc, new_m, 1.0f);
                simd::s_store(&l[i], exp_diff * simd::s_load(&l[i]) + rowsum);

                if (exp_diff != 1.0f) {
                    T* orow = o_acc + i * Dv;
                    auto v_ed = simd::v_set1(orow, exp_diff);
                    int64_t d = 0;
                    for (; d + L <= Dv; d += L) {
                        simd::v_store(orow + d, simd::v_mul(simd::v_load(orow + d), v_ed));
                    }
                    for (; d < Dv; ++d) {
                        simd::s_store(&orow[d], simd::s_load(&orow[d]) * exp_diff);
                    }
                }
            }

            // ---- o_acc += S @ V[j0:j0+Bc] --------------------------
            for (int64_t k0 = 0; k0 < actual_bc; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, actual_bc - k0));
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                    / static_cast<int>(sizeof(T)));
                // V rows [j0+k0, +actual_kc) x Dv, packed once for the whole
                // Br-row block.
                tile_pack_rhs(false, static_cast<int>(Dv), actual_kc,
                                      pack_b, ldd_b,
                                      vb + (koff + k0) * v_step, static_cast<int>(v_step),
                                      1.0f);
                tile_mma_direct(actual_br, static_cast<int>(Dv), actual_kc,
                                        o_acc, static_cast<int>(Dv),
                                        s + k0, actual_bc,
                                        pack_b, /*ldb=*/-1, /*stage=*/nullptr, kNegInf, kPosInf, first_tile && k0 == 0);
            }
            first_tile = false;
            j0 += actual_bc;
        }

        // ---- final normalize: output = o_acc / l -------------------
        for (int i = 0; i < actual_br; ++i) {
            const float l_i = simd::s_load(&l[i]);
            // A row every tile masked has l == 0; its output is zeros.
            const float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
            const T* src = o_acc + i * Dv;
            T* dst = o_head + (i0 + i) * o_rs;
            auto v_inv = simd::v_set1(src, inv_l);
            int64_t d = 0;
            for (; d + L <= Dv; d += L) {
                simd::v_store(dst + d, simd::v_mul(simd::v_load(src + d), v_inv));
            }
            for (; d < Dv; ++d) {
                simd::s_store(&dst[d], simd::s_load(&src[d]) * inv_l);
            }
        }
    };

    ctx.cpu.run(0, NG, run);
}

// ---- single-row decode kernel (Sq == 1) ----------------------------------
//
// With one query row both products collapse to GEMV, so K and V are streamed
// once each: no B packing, no M == 1 MMA tiles, and no materialized score
// matrix beyond the row itself. Scores and the output accumulator are f32
// whatever the element type — f16 converts once per loaded vector instead of
// accumulating in f16, which also keeps the deviation inside kF16AccumTol.
//
// One task per (batch, head). A task reads Sk·Dqk elements of K and Sk·Dv of
// V, the same traffic the packed route spends on packing alone before its
// MMA sees a byte.

inline simd::v_f32x8 vload_f32x8(const float* p) { return simd::v_load(p); }
inline simd::v_f32x8 vload_f32x8(const half* p)  { return simd::v_cvt_f16_to_f32(p); }

inline void vstore_f32x8(float* p, const simd::v_f32x8& v) { simd::v_store(p, v); }
inline void vstore_f32x8(half* p, const simd::v_f32x8& v)  { simd::v_store(p, simd::v_cvt_f32_to_f16(v)); }

/// Dot product of two n-wide rows, f32 accumulation.
template <class T>
inline float dot_row(const T* a, const T* b, int64_t n)
{
    constexpr int L = simd::simd_lane_for<T>;
    auto acc = simd::v_zero_f32x8();
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        acc = simd::v_fmadd(vload_f32x8(a + i), vload_f32x8(b + i), acc);
    }
    float s = simd::v_reduce_sum(acc);
    for (; i < n; ++i) { s += simd::s_load(&a[i]) * simd::s_load(&b[i]); }
    return s;
}

/// Four n-wide dot products against one query row, written to @p out. A
/// horizontal reduce costs about as much as the 16 vector FMAs of a 128-wide
/// row, so four keys per reduce is what keeps the GEMV off the latency floor.
template <class T>
inline void dot_row4(const T* q, const T* k0, const T* k1, const T* k2, const T* k3,
                     int64_t n, float* out)
{
    constexpr int L = simd::simd_lane_for<T>;
    auto a0 = simd::v_zero_f32x8();
    auto a1 = a0, a2 = a0, a3 = a0;
    int64_t i = 0;
    for (; i + L <= n; i += L) {
        auto qv = vload_f32x8(q + i);
        a0 = simd::v_fmadd(qv, vload_f32x8(k0 + i), a0);
        a1 = simd::v_fmadd(qv, vload_f32x8(k1 + i), a1);
        a2 = simd::v_fmadd(qv, vload_f32x8(k2 + i), a2);
        a3 = simd::v_fmadd(qv, vload_f32x8(k3 + i), a3);
    }
    float s0 = simd::v_reduce_sum(a0);
    float s1 = simd::v_reduce_sum(a1);
    float s2 = simd::v_reduce_sum(a2);
    float s3 = simd::v_reduce_sum(a3);
    for (; i < n; ++i) {
        const float qs = simd::s_load(&q[i]);
        s0 += qs * simd::s_load(&k0[i]);
        s1 += qs * simd::s_load(&k1[i]);
        s2 += qs * simd::s_load(&k2[i]);
        s3 += qs * simd::s_load(&k3[i]);
    }
    out[0] = s0; out[1] = s1; out[2] = s2; out[3] = s3;
}

template <class T>
void attention_decode_impl(const AttentionConfig& cfg,
                           TensorView& output,
                           std::span<const TensorView> inputs,
                           const ComputeContext& ctx,
                           const AttentionPlan& plan,
                           void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = cfg.v_is_k ? inputs[1] : inputs[2];
    const bool split = has_past_view(cfg, inputs);
    const auto& Kp = split ? inputs[past_slot(cfg)] : K;
    const auto& Vp = split ? inputs[past_slot(cfg) + (cfg.v_is_k ? 0 : 1)] : V;

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Dqk = hs.Dqk, Dv = hs.Dv;
    const int64_t Sk_new = hs.Sk, Sk_past = hs.Sk_past;
    const int64_t Sk = Sk_new + Sk_past;   // the whole [past | new] run
    const bool merged = hs.merged;
    NNOPS_ASSERT(Sq == 1);

    const float scale = (cfg.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* kp_ptr = split ? Kp.ptr<T>() : nullptr;
    const T* vp_ptr = split ? Vp.ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t kp_rs = split ? Kp.row_stride_elems() : k_rs;
    const int64_t vp_rs = split ? Vp.row_stride_elems() : v_rs;
    const int64_t o_rs = output.row_stride_elems();
    const int64_t v_head_stride = merged_v_head_stride(cfg, Dqk, Dv);

    const int64_t per_slot = plan.decode_elems;   // f32: [scores:Sk][o_acc:Dv]
    const int64_t NG = B * H;
    float* ws = static_cast<float*>(workspace);

    // Bottom-right alignment makes the limit Sk — the row attends to every
    // cached key; an explicit position_offset can cut it shorter.
    const int64_t n_valid = causal_limit(cfg, 0, Sq, Sk);

    const auto run = [&](int64_t idx) {
        const int64_t b = idx / H;
        const int64_t h = idx % H;
        const int64_t hkv = h / (H / Hkv);

        const T* q_row;
        const T* k_head;
        const T* v_head;
        T* o_row;
        if (merged) {
            q_row  = q_ptr + b * q_rs + h * Dqk;
            k_head = k_ptr + b * Sk_new * k_rs + hkv * Dqk;
            v_head = v_ptr + b * hs.Sv * v_rs + hkv * v_head_stride;
            o_row  = out_ptr + b * o_rs + h * Dv;
        } else {
            q_row  = q_ptr + (b * H + h) * q_rs;
            k_head = k_ptr + (b * Hkv + hkv) * Sk_new * k_rs;
            v_head = v_ptr + (b * Hkv + hkv) * hs.Sv * v_rs;
            o_row  = out_ptr + (b * H + h) * o_rs;
        }

        // [past | new], the past segment empty when no past is passed.
        const T* k_base[2] = {k_head, k_head};
        const T* v_base[2] = {v_head, v_head};
        int64_t k_str[2] = {k_rs, k_rs};
        int64_t v_str[2] = {v_rs, v_rs};
        int64_t seg_lo[2] = {0, 0};
        int64_t seg_hi[2] = {0, Sk};
        if (split) {
            k_base[0] = merged ? kp_ptr + b * Sk_past * kp_rs + hkv * Dqk
                               : kp_ptr + (b * Hkv + hkv) * Sk_past * kp_rs;
            v_base[0] = merged ? vp_ptr + b * Sk_past * vp_rs + hkv * v_head_stride
                               : vp_ptr + (b * Hkv + hkv) * Sk_past * vp_rs;
            k_str[0] = kp_rs;
            v_str[0] = vp_rs;
            seg_lo[1] = Sk_past;
            seg_hi[0] = Sk_past;
        }

        const int64_t slot = scratch_slot_of(ctx.cpu, NG, idx);
        float* scores = ws + slot * per_slot;
        float* o_acc  = scores + Sk;

        // ---- scores = q · K^T ----------------------------------------
        // The scale rides through the softmax temperature instead of touching
        // the row, so this pass stays a pure dot product. Nothing past the
        // row's causal limit is scored.
        for (int s = 0; s < 2; ++s) {
            const int64_t hi = std::min(seg_hi[s], n_valid);
            const T* kb = k_base[s];
            const int64_t rs = k_str[s];
            int64_t t = seg_lo[s];
            for (; t + 4 <= hi; t += 4) {
                const T* kk = kb + (t - seg_lo[s]) * rs;
                dot_row4(q_row, kk, kk + rs, kk + 2 * rs, kk + 3 * rs, Dqk, scores + t);
            }
            for (; t < hi; ++t) {
                scores[t] = dot_row(q_row, kb + (t - seg_lo[s]) * rs, Dqk);
            }
        }

        // ---- softmax over the valid prefix (already normalized) ------
        if (n_valid <= 0) {
            for (int64_t d = 0; d < Dv; ++d) { simd::s_store(&o_row[d], 0.0f); }
            return;
        }
        kernel::softmax_process_standard_row<float>(scores, scores, n_valid,
                                                    /*log_softmax=*/false, scale);

        // ---- o = P · V, f32 accumulator ------------------------------
        // Four value rows per accumulator pass: the o_acc round-trip drops 4x
        // and the four FMAs into one lane vector still leave Dv/8 independent
        // chains. A row that softmaxed to zero contributes a zero, so the
        // branch-free path costs nothing and drops a per-row test.
        for (int64_t d = 0; d < Dv; ++d) { o_acc[d] = 0.0f; }
        constexpr int64_t lanes = simd::simd_default_lane_f32;
        for (int s = 0; s < 2; ++s) {
            const int64_t hi = std::min(seg_hi[s], n_valid);
            const T* vb = v_base[s];
            const int64_t rs = v_str[s];
            int64_t t = seg_lo[s];
            for (; t + 4 <= hi; t += 4) {
                const T* r0 = vb + (t - seg_lo[s]) * rs;
                const T* r1 = r0 + rs;
                const T* r2 = r1 + rs;
                const T* r3 = r2 + rs;
                const float p0 = scores[t + 0], p1 = scores[t + 1];
                const float p2 = scores[t + 2], p3 = scores[t + 3];
                const auto vp0 = simd::v_set1_f32x8(p0), vp1 = simd::v_set1_f32x8(p1);
                const auto vp2 = simd::v_set1_f32x8(p2), vp3 = simd::v_set1_f32x8(p3);
                int64_t d = 0;
                for (; d + lanes <= Dv; d += lanes) {
                    auto acc = simd::v_load(o_acc + d);
                    acc = simd::v_fmadd(vp0, vload_f32x8(r0 + d), acc);
                    acc = simd::v_fmadd(vp1, vload_f32x8(r1 + d), acc);
                    acc = simd::v_fmadd(vp2, vload_f32x8(r2 + d), acc);
                    acc = simd::v_fmadd(vp3, vload_f32x8(r3 + d), acc);
                    simd::v_store(o_acc + d, acc);
                }
                for (; d < Dv; ++d) {
                    o_acc[d] += p0 * simd::s_load(&r0[d]) + p1 * simd::s_load(&r1[d])
                              + p2 * simd::s_load(&r2[d]) + p3 * simd::s_load(&r3[d]);
                }
            }
            for (; t < hi; ++t) {
                const float p = scores[t];
                const T* vrow = vb + (t - seg_lo[s]) * rs;
                auto vp = simd::v_set1_f32x8(p);
                int64_t d = 0;
                for (; d + lanes <= Dv; d += lanes) {
                    auto acc = simd::v_load(o_acc + d);
                    acc = simd::v_fmadd(vp, vload_f32x8(vrow + d), acc);
                    simd::v_store(o_acc + d, acc);
                }
                for (; d < Dv; ++d) { o_acc[d] += p * simd::s_load(&vrow[d]); }
            }
        }

        // ---- store ---------------------------------------------------
        int64_t d = 0;
        for (; d + simd::simd_default_lane_f32 <= Dv; d += simd::simd_default_lane_f32) {
            vstore_f32x8(o_row + d, simd::v_load(o_acc + d));
        }
        for (; d < Dv; ++d) { simd::s_store(&o_row[d], o_acc[d]); }
    };

    ctx.cpu.run(0, NG, run);
}

}  // anonymous namespace

// =========================================================================
//  Plan — single source of truth for tile sizes + workspace layout
// =========================================================================

namespace {

/// Slots a plan must size, matching scratch_slot_count's choice at run time.
inline int64_t plan_slot_count(int num_threads, bool use_thread_slots, int64_t tasks)
{
    const int64_t nt = std::max<int64_t>(num_threads, 1);
    if (use_thread_slots && nt < tasks) { return nt; }
    return nt <= 1 ? 1 : tasks;
}

/// Per-dtype plan resolution, shared by get_attention_plan's dispatch below.
/// f32 and f16 differ in NR, Kc and element size, exactly as in matmul.
template <class T>
AttentionPlan attention_plan_for(const AttentionConfig& cfg,
                                 std::span<const TensorDesc> inputs,
                                 int num_threads,
                                 bool use_thread_slots)
{
    const auto& q = inputs[0];
    const auto& k = inputs[1];

    AttentionPlan plan;

    const int64_t H = cfg.num_heads;
    const bool merged = (q.rank == 3);
    int64_t B, Sq, Sk_new, Dqk;
    if (merged) {
        B  = q.dims[0];
        Sq = q.dims[1];
        Sk_new = k.dims[1];
        Dqk = q.dims[2] / H;
    } else {
        B  = q.dims[0];
        Sq = q.dims[2];
        Sk_new = k.dims[2];
        Dqk = q.dims[3];
    }
    const int64_t Dv = cfg.v_is_k ? cfg.v_head_dim : Dqk;

    // A past segment makes the run [past | new]; every route is sized by the
    // whole run, and the standard route — which packs B from one base pointer
    // — is skipped in favour of flash.
    const size_t pslot = past_slot(cfg);
    const bool split = inputs.size() > pslot && inputs[pslot].rank != 0;
    const int64_t Sk_past = split ? inputs[pslot].dims[merged ? 1 : 2] : 0;
    const int64_t Sk = Sk_new + Sk_past;

    // Decode — one query row and no mask: both products are GEMV, so the
    // packing-dominated tiled routes are pure overhead. An optional mask
    // arrives as a rank-0 descriptor and keeps the standard route.
    const size_t mslot = mask_slot(cfg);
    const bool has_mask = inputs.size() > mslot && inputs[mslot].rank != 0;
    plan.use_decode = (Sq == 1) && !has_mask;
    if (plan.use_decode) {
        plan.decode_elems = Sk + Dv;   // f32: scores row then output accumulator
        plan.num_slots = plan_slot_count(num_threads, use_thread_slots, B * H);
        plan.workspace_size = static_cast<size_t>(plan.num_slots)
                            * static_cast<size_t>(plan.decode_elems) * sizeof(float);
        return plan;
    }

    plan.use_flash = split || should_use_flash_attention(Sq, Sk);

    // Flash path — sized with the same Br/Bc the kernel uses. Slots are
    // per worker thread when thread ids are reported, else per query block.
    if (plan.use_flash) {
        int Br, Bc;
        resolve_flash_tile_sizes(sizeof(T), Sq, Sk, Dqk, Dv, cfg.causal, Br, Bc);
        plan.Br = Br;
        plan.Bc = Bc;
        const int64_t per_task = flash_per_task_elems<T>(Br, Bc, Dqk, Dv);
        const int64_t NQ = (Sq + Br - 1) / Br;
        plan.num_slots = plan_slot_count(num_threads, use_thread_slots, B * H * NQ);
        plan.workspace_size = static_cast<size_t>(plan.num_slots)
                            * static_cast<size_t>(per_task) * sizeof(T);
        return plan;
    }

    constexpr int mr_max = std::is_same_v<T, half> ? MR_MAX_F16 : MR_MAX_F32;
    constexpr int nr_max = nr_max_flt<T>();
    constexpr int kc     = static_cast<int>(kc_for<T>());
    constexpr int esz    = static_cast<int>(sizeof(T));

    int mc1, nc1;
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(Sk),
                       mr_max, nr_max, kc, esz, num_threads, mc1, nc1);
    int mc2, nc2;
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(Dv),
                       mr_max, nr_max, kc, esz, num_threads, mc2, nc2);
    plan.mc1 = mc1; plan.nc1 = nc1;
    plan.mc2 = mc2; plan.nc2 = nc2;

    // One packed-B slot for both GEMMs: stride from the longer k-block, panels
    // from the wider n-block.
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(
            kc_pack_bound<T>(std::max(Dqk, Sk))) * esz) / esz);
    const int np_full = std::max(panels_for_pack<T>(nc1),
                                 panels_for_pack<T>(std::min<int64_t>(Dv, nc2)));

    const int64_t scores_elems = Sq * Sk;
    const int64_t pack_elems   = static_cast<int64_t>(np_full) * ldd_b_full;
    const int64_t per_head     = scores_elems + pack_elems;

    plan.num_slots = plan_slot_count(num_threads, use_thread_slots, B * H);
    plan.workspace_size = static_cast<size_t>(plan.num_slots)
                        * static_cast<size_t>(per_head) * sizeof(T);
    return plan;
}

}  // namespace

AttentionPlan get_attention_plan(const AttentionConfig& cfg,
                                 std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> /*outputs*/,
                                 int num_threads,
                                 bool use_thread_slots)
{
    NNOPS_ASSERT(inputs.size() >= 2);

    const DataType dt = inputs[0].dtype;

    // f32 and f16 take the tiled path; any other dtype, or a Q/K dtype
    // mismatch, returns a zeroed plan.
    if (inputs[1].dtype != dt) { return AttentionPlan{}; }
    if (!cfg.v_is_k && inputs.size() > 2 && inputs[2].dtype != dt) {
        return AttentionPlan{};
    }
    if (dt == DataType::f16) {
        return attention_plan_for<half>(cfg, inputs, num_threads, use_thread_slots);
    }
    if (dt == DataType::f32) {
        return attention_plan_for<float>(cfg, inputs, num_threads, use_thread_slots);
    }
    return AttentionPlan{};
}

// =========================================================================
//  Public API
// =========================================================================

Status attention_cpu(const AttentionConfig& cfg,
                     TensorView& output,
                     std::span<const TensorView> inputs,
                     const ComputeContext& ctx,
                     void* /*workspace*/)
{
    const size_t required = cfg.v_is_k ? 2u : 3u;
    if (inputs.size() < required) { return Status::invalid_argument; }

    const DataType dt = inputs[0].data_type();

    bool dtypes_ok = (dt == DataType::f32 || dt == DataType::f16) &&
                     output.data_type() == dt;
    for (size_t i = 1; i < inputs.size() && dtypes_ok; ++i) {
        if (inputs[i].is_empty()) { continue; }   // absent optional mask
        dtypes_ok = inputs[i].data_type() == dt;
    }
    if (!dtypes_ok) { return Status::unsupported; }

    // One plan decides tiles and scratch, and the scratch is allocated once
    // here, outside the parallel dispatch — no per-task allocation. Slots are
    // indexed by worker id, so the count is the workers, not the grid: an
    // over-subscribed provider would otherwise allocate 4x the scratch.
    detail::SmallVector<TensorDesc, 16> descs;
    for (size_t i = 0; i < inputs.size(); ++i) { descs.push_back(inputs[i].desc()); }
    const TensorDesc outs[] = {output.desc()};
    const AttentionPlan plan = get_attention_plan(cfg, descs, outs,
                                                  ctx.cpu.worker_count(),
                                                  ctx.cpu.thread_slot_count() > 0);
    PoolPtr scratch(plan.workspace_size);

    if (dt == DataType::f16) {
        if (plan.use_decode) {
            attention_decode_impl<half>(cfg, output, inputs, ctx, plan, scratch.as<float>());
        } else if (plan.use_flash) {
            attention_flash_impl<half>(cfg, output, inputs, ctx, plan, scratch.as<half>());
        } else {
            attention_impl<half>(cfg, output, inputs, ctx, plan, scratch.as<half>());
        }
    } else {
        if (plan.use_decode) {
            attention_decode_impl<float>(cfg, output, inputs, ctx, plan, scratch.as<float>());
        } else if (plan.use_flash) {
            attention_flash_impl<float>(cfg, output, inputs, ctx, plan, scratch.as<float>());
        } else {
            attention_impl<float>(cfg, output, inputs, ctx, plan, scratch.as<float>());
        }
    }
    return Status::ok;
}

// =========================================================================
//  Operator entry points
// =========================================================================

Status multi_head_attention_cpu(const MultiHeadAttentionAttributes& attrs,
                                TensorView& output,
                                std::span<const TensorView> inputs,
                                const ComputeContext& ctx,
                                void* workspace)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.scale = attrs.scale;
    cfg.causal = attrs.causal;
    cfg.position_offset = attrs.position_offset;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

Status group_query_attention_cpu(const GroupQueryAttentionAttributes& attrs,
                                 TensorView& output,
                                 std::span<const TensorView> inputs,
                                 const ComputeContext& ctx,
                                 void* workspace)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.num_kv_heads = attrs.kv_num_heads;
    cfg.scale = attrs.scale;
    cfg.causal = attrs.causal;
    cfg.position_offset = attrs.position_offset;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

Status multi_latent_attention_cpu(const MultiLatentAttentionAttributes& attrs,
                                  TensorView& output,
                                  std::span<const TensorView> inputs,
                                  const ComputeContext& ctx,
                                  void* workspace)
{
    AttentionConfig cfg;
    cfg.num_heads = attrs.num_heads;
    cfg.num_kv_heads = attrs.kv_num_heads;
    cfg.v_head_dim = attrs.v_head_dim > 0 ? attrs.v_head_dim : attrs.kv_lora_rank;
    cfg.scale = attrs.scale != 0.0f
        ? attrs.scale
        : 1.0f / std::sqrt(static_cast<float>(attrs.qk_nope_head_dim + attrs.qk_rope_head_dim));
    cfg.causal = attrs.causal;
    cfg.position_offset = attrs.position_offset;
    cfg.v_is_k = true;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
