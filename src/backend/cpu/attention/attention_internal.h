#pragma once
/// @file attention_internal.h
/// @brief Machinery the attention routes and their plan share.
///
/// attention.h describes the routes, the head mapping and the scratch contract;
/// this header is the implementation side of that contract. It is included only
/// by the attention translation units, which are split by route:
///
///   attention_standard.cpp  — materialized scores, Q·Kᵀ → softmax → ·V
///   attention_flash.cpp     — FlashAttention-1 online softmax over KV tiles
///   attention_decode.cpp    — the Sq == 1 GEMV route, head batching + KV split
///   attention_plan.cpp      — get_attention_plan: tile sizes and scratch bytes
///   attention.cpp           — attention_cpu and the three operator entries
///
/// What lives here is what those files must agree on: the dtype tiling
/// constants, head-shape extraction, the scratch-slot rule (a kernel's slot
/// index and the plan's slot count are one decision), the causal row limit, and
/// the route geometry both the plan and the kernels size themselves from.

#include "attention.h"
#include "backend/cpu/matmul/matmul_helper.h"           // tile_mma_direct / tile_pack_rhs + panel constants
#include "backend/cpu/simd_kernel/simd_softmax.hpp"     // softmax + FlashAttention row kernels
#include "nnops/detail/assert.hpp"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace nnops::backend::cpu {

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

// ---- per-batch key-run length (seqlens_k) --------------------------------

/// The optional per-batch run length, the last positional slot: Q, K, V,
/// [mask], [past K], [past V], [seqlens_k] — or Q, KV, [mask], [past KV],
/// [seqlens_k] when v_is_k. An int32 [B] tensor whose entry is the number of
/// valid keys in the whole [past | new] run for that batch; the valid keys are
/// that prefix, so right padding is truncated and each batch may sit at its own
/// position. Absent, every batch runs to the tensor's Sk.
inline size_t seqlens_slot(const AttentionConfig& cfg) { return cfg.v_is_k ? 4u : 6u; }

inline bool has_seqlens_input(const AttentionConfig& cfg, std::span<const TensorView> inputs)
{
    const size_t s = seqlens_slot(cfg);
    return inputs.size() > s && !inputs[s].is_empty();
}

/// Batch @p b's valid key-run length, clamped to the tensor's run.
inline int64_t run_length(const int32_t* seqlens, int64_t b, int64_t Sk)
{
    if (seqlens == nullptr) { return Sk; }
    const int64_t v = seqlens[b];
    if (v <= 0) { return 0; }
    return v < Sk ? v : Sk;
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

inline int64_t scratch_slot_of(const CpuBackend& cpu, int64_t tasks, int64_t idx)
{
    if (scratch_per_worker(cpu, tasks)) { return cpu.current_thread_id(); }
    return cpu.worker_count() <= 1 ? 0 : idx;
}

// ---- row limit -----------------------------------------------------------

/// Exclusive end of query row @p i's valid key range: the batch's whole run
/// when not causal, else q0 + i + 1 with q0 the row-0 position —
/// cfg.position_offset when given, bottom-right (run - Sq) otherwise. @p run is
/// the batch's valid key-run length (run_length), so a shorter run truncates
/// both the padding and the causal diagonal.
inline int64_t causal_limit(const AttentionConfig& cfg, int64_t i,
                            int64_t Sq, int64_t run)
{
    if (!cfg.causal) { return run; }
    const int64_t q0 = cfg.position_offset >= 0 ? cfg.position_offset : (run - Sq);
    const int64_t lim = q0 + i + 1;
    if (lim < 0) { return 0; }
    return lim > run ? run : lim;
}

template <class T>
inline void row_fill(T* row, int64_t n, float val)
{
    for (int64_t i = 0; i < n; ++i) { simd::s_store(&row[i], val); }
}

// ---- flash route geometry ------------------------------------------------
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

// ---- decode route geometry -----------------------------------------------
// Shared by attention_decode.cpp and the plan: the score block size the kernel
// walks the run in, the head batch one task serves, and the KV-split caps.

constexpr int64_t kDecodeChunk = 1024;   ///< keys per decode softmax block
constexpr int     kDecodeHeadBatch = 8;  ///< query heads one decode task serves
constexpr int     kDecodeMaxSplits = 16; ///< KV slices per decode unit, at most
constexpr int64_t kDecodeMinSlice = 256; ///< shortest slice worth its merge pass

// ---- route entries -------------------------------------------------------
// Each route file defines one; the dispatch in attention.cpp calls it after
// validating the dtype. Scratch is the pooled workspace from the plan.

void attention_standard_cpu(const AttentionConfig& cfg,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            const AttentionPlan& plan,
                            void* scratch);

void attention_flash_cpu(const AttentionConfig& cfg,
                         TensorView& output,
                         std::span<const TensorView> inputs,
                         const ComputeContext& ctx,
                         const AttentionPlan& plan,
                         void* scratch);

void attention_decode_cpu(const AttentionConfig& cfg,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const AttentionPlan& plan,
                          void* scratch);

}  // namespace nnops::backend::cpu
