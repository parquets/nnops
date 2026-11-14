/// @file attention_plan.cpp
/// @brief get_attention_plan — the single source of truth for tile sizes and
///        the workspace layout.
///
/// The plan resolves the route (decode / flash / standard) from the descriptor
/// shapes alone, then sizes that route's tiles and scratch bytes with the same
/// geometry constants its kernel uses. attention_cpu allocates once from
/// workspace_size, outside the parallel dispatch, and each route entry reads the
/// tile fields back — a plan and a kernel that disagree is exactly the bug this
/// file exists to prevent.
///
/// Slots are sized per worker thread when thread ids are reported, else per
/// task; see scratch_per_worker in attention_internal.h.

#include "attention_internal.h"

namespace nnops::backend::cpu {

namespace {

/// Slots a plan must size, matching scratch_per_worker's choice at run time.
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
    const int64_t Dv = cfg.v_head_dim > 0 ? cfg.v_head_dim : Dqk;

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
        // Head batching: a GQA/MQA group's query heads share one KV head, so
        // one task streams the run for the whole group. The batch is capped:
        // past eight heads the score tile and the live accumulators push the
        // GEMV out of registers for a read that is already amortized.
        const int64_t Hkv = cfg.kv_heads();
        NNOPS_ASSERT(H % Hkv == 0);
        const int64_t G = H / Hkv;
        const int64_t hb = std::min<int64_t>(G, kDecodeHeadBatch);
        const int64_t nb = (G + hb - 1) / hb;
        const int64_t units = B * Hkv * nb;

        // KV splitting: batching shrinks the task grid by up to eight, so when
        // it no longer covers the workers the run is cut into slices and the
        // partial softmax states merged. A slice below kDecodeMinSlice does
        // not amortize its own max/merge pass.
        int64_t splits = 1;
        const int64_t nt = std::max<int64_t>(num_threads, 1);
        if (nt > 1 && units < nt) {
            splits = std::min<int64_t>(kDecodeMaxSplits, (nt + units - 1) / units);
            while (splits > 1 && (Sk + splits - 1) / splits < kDecodeMinSlice) {
                --splits;
            }
        }

        plan.decode_heads = static_cast<int>(hb);
        plan.decode_splits = static_cast<int>(splits);
        // f32 per slot: [scores: hb][chunk][o_acc: hb][Dv][max: hb][sum: hb].
        // The score block is a fixed chunk, not the whole run, so the scratch
        // stays flat as the cache grows.
        plan.decode_elems = hb * kDecodeChunk + hb * Dv + 2 * hb;
        const int64_t tasks = units * splits;
        // Both dispatches draw slots from the same pool: the slice kernel and
        // the (batch, query head) merge pass after it.
        plan.num_slots = plan_slot_count(num_threads, use_thread_slots,
                                         std::max(tasks, B * H));
        size_t bytes = static_cast<size_t>(plan.num_slots)
                     * static_cast<size_t>(plan.decode_elems) * sizeof(float);
        if (splits > 1) {
            // One partial per (task, batched head): output, max, sum.
            bytes += static_cast<size_t>(tasks) * static_cast<size_t>(hb)
                   * static_cast<size_t>(Dv + 2) * sizeof(float);
        }
        plan.workspace_size = bytes;
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

}  // anonymous namespace

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
#ifdef NNOPS_ARCH_AARCH64
    if (dt == DataType::bf16) {
        return attention_plan_for<bfloat16>(cfg, inputs, num_threads, use_thread_slots);
    }
#endif
    if (dt == DataType::f32) {
        return attention_plan_for<float>(cfg, inputs, num_threads, use_thread_slots);
    }
    return AttentionPlan{};
}

}  // namespace nnops::backend::cpu
