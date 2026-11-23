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
    if (use_thread_slots && nt < tasks) {
        return nt;
    }
    return nt <= 1 ? 1 : tasks;
}

/// Per-dtype plan resolution, shared by get_attention_plan's dispatch below.
/// f32 and f16 differ in NR, Kc and element size, exactly as in matmul.
template <class T>
AttentionPlan attention_plan_for(const AttentionConfig& cfg, std::span<const TensorDesc> inputs,
                                 int num_threads, bool use_thread_slots)
{
    const auto& q = inputs[0];
    const auto& k = inputs[1];

    AttentionPlan plan;

    const int64_t H = cfg.num_heads;
    const HeadLayout q_layout = (q.rank == 3) ? HeadLayout::Merged :
                                cfg.bshd      ? HeadLayout::BSHD :
                                                HeadLayout::BHSD;
    const HeadLayout kv_layout = (k.rank == 3) ? HeadLayout::Merged :
                                 cfg.kv_bshd   ? HeadLayout::BSHD :
                                                 HeadLayout::BHSD;
    const int64_t sd = head_seq_dim(q_layout);
    const int64_t ksd = head_seq_dim(kv_layout);
    const int64_t B = q.dims[0];
    const int64_t Sq = q.dims[static_cast<size_t>(sd)];
    const int64_t Sk_new = k.dims[static_cast<size_t>(ksd)];
    const int64_t Dqk = (q_layout == HeadLayout::Merged) ? q.dims[2] / H : q.dims[3];
    const int64_t Dv = cfg.v_head_dim > 0 ? cfg.v_head_dim : Dqk;
    const bool kv8 = k.dtype == DataType::s8;

    // A past segment makes the run [past | new]; every route is sized by the
    // whole run, and the standard route — which packs B from one base pointer
    // — is skipped in favour of flash. The descriptor has no data, so a
    // past_sequence_length narrowing is not visible here: the plan sizes the
    // past buffer's full row count, and run_length clamps the reads.
    const bool split = cfg.past_k_slot >= 0 && inputs.size() > static_cast<size_t>(cfg.past_k_slot) &&
                       inputs[static_cast<size_t>(cfg.past_k_slot)].rank != 0;
    const int64_t Sk_past = split ? inputs[static_cast<size_t>(cfg.past_k_slot)]
                                        .dims[static_cast<size_t>(ksd)] :
                                    0;
    const int64_t Sk = Sk_new + Sk_past;

    // Decode — one query row and no mask: both products are GEMV, so the
    // packing-dominated tiled routes are pure overhead. A key_padding_mask
    // that only carries lengths is not a mask and keeps the decode route.
    const bool has_bias = cfg.bias_slot >= 0 && inputs.size() > static_cast<size_t>(cfg.bias_slot) &&
                          inputs[static_cast<size_t>(cfg.bias_slot)].rank != 0;
    const bool has_keep = cfg.kpm_slot >= 0 && inputs.size() > static_cast<size_t>(cfg.kpm_slot) &&
                          inputs[static_cast<size_t>(cfg.kpm_slot)].rank >= 2;
    const bool has_mask = has_bias || has_keep;
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
        plan.num_slots = plan_slot_count(num_threads, use_thread_slots, std::max(tasks, B * H));
        size_t bytes = static_cast<size_t>(plan.num_slots) *
                       static_cast<size_t>(plan.decode_elems) * sizeof(float);
        if (splits > 1) {
            // One partial per (task, batched head): output, max, sum.
            bytes += static_cast<size_t>(tasks) * static_cast<size_t>(hb) *
                     static_cast<size_t>(Dv + 2) * sizeof(float);
        }
        plan.workspace_size = bytes;
        return plan;
    }

    plan.use_flash = split || should_use_flash_attention(Sq, Sk);

    // Flash path — sized with the same Br/Bc the kernel uses. Slots are
    // per worker thread when thread ids are reported, else per query block.
    if (plan.use_flash) {
        // Which engine each product takes is settled before the tiles, because
        // an integer GEMM takes the wider causal pair (see the caps). qk8
        // reads Dqk alone; pv8 reads Bc only through the one-k-block cap,
        // which every Bc resolve_flash_tile_sizes can return already meets. The
        // integer GEMMs' extra scratch (packed query/probability blocks, the
        // query's row scales, the int32 accumulator tile) then rides in every
        // slot, so the head-batch and split rules below see the same per-task
        // size the kernel uses.
        const bool qk8 = qk_i8_enabled(cfg, kv8, Dqk);
        const bool pv8 = pv_i8_enabled(cfg, kv8, kFlashBcMax);
        int Br, Bc;
        resolve_flash_tile_sizes(sizeof(T), Sq, Sk, Dqk, Dv, cfg.causal, qk8 || pv8, Br, Bc);
        plan.Br = Br;
        plan.Bc = Bc;
        const int64_t NQ = (Sq + Br - 1) / Br;
        plan.flash_qk_i8 = qk8;
        plan.flash_pv_i8 = pv8;

        // GQA head batching: the group's query heads share one K/V head, so one
        // task can pack and stream each K/V tile once for all of them. Only do
        // it while the reduced grid still leaves several waves for the workers
        // — batching near the worker count turns a balanced static share into
        // one short final wave — and the extra per-head scratch stays inside
        // the pool budget.
        const int64_t Hkv = cfg.kv_heads();
        const int64_t G = H / Hkv;
        const int64_t tasks_unbatched = B * H * NQ;
        const int64_t nt = std::max<int64_t>(num_threads, 1);
        const int64_t min_tasks = 8 * nt;
        int hb = 1;
        if (G > 1) {
            if (tasks_unbatched >= min_tasks) {
                const int cap = static_cast<int>(std::min<int64_t>(G, kFlashHeadBatchMax));
                for (int cand = cap; cand >= 2; --cand) {
                    const int64_t nb = (G + cand - 1) / cand;
                    const int64_t tasks = B * Hkv * nb * NQ;
                    if (tasks < min_tasks) {
                        continue;
                    }
                    const int64_t per_task =
                        flash_per_task_elems<T>(Br, Bc, Dqk, Dv, cand, kv8, qk8, pv8);
                    const int64_t slots =
                        plan_slot_count(num_threads, use_thread_slots, tasks);
                    if (static_cast<size_t>(slots) * static_cast<size_t>(per_task) *
                            sizeof(T) >
                        kFlashScratchBudget) {
                        continue;
                    }
                    hb = cand;
                    break;
                }
            }
        }
        plan.flash_heads = hb;
        const int64_t nb = (G + hb - 1) / hb;
        const int64_t units = B * Hkv * nb * NQ;

        // KV splitting: a query-block grid smaller than the worker count leaves
        // workers idle (a short chunk against a long cache, or a low head
        // count). Cut the run into slices, each an independent task, and merge
        // the unnormalized online-softmax states afterwards.
        int64_t splits = 1;
        if (nt > 1 && units < nt) {
            splits = std::min<int64_t>(kFlashMaxSplits, (4 * nt + units - 1) / units);
            const int64_t min_slice = std::max<int64_t>(Bc, kFlashMinSlice);
            while (splits > 1 && (Sk + splits - 1) / splits < min_slice) {
                --splits;
            }
            const int64_t task_elems = flash_per_task_elems<T>(Br, Bc, Dqk, Dv, hb, kv8, qk8, pv8);
            while (splits > 1) {
                const int64_t tasks = units * splits;
                const int64_t slots =
                    plan_slot_count(num_threads, use_thread_slots, std::max(tasks, B * H * NQ));
                const size_t slot_bytes =
                    static_cast<size_t>(slots) * static_cast<size_t>(task_elems) * sizeof(T);
                const size_t partial_bytes =
                    static_cast<size_t>(tasks) * static_cast<size_t>(hb) *
                    static_cast<size_t>(Br) * static_cast<size_t>(Dv + 2) * sizeof(float);
                if (align_up_bytes(slot_bytes, 16) + partial_bytes +
                        static_cast<size_t>(slots) * static_cast<size_t>(Dv) * sizeof(float) <=
                    kFlashScratchBudget) {
                    break;
                }
                --splits;
            }
        }
        plan.flash_splits = static_cast<int>(splits);
        const int64_t tasks = units * splits;
        const int64_t per_task = flash_per_task_elems<T>(Br, Bc, Dqk, Dv, hb, kv8, qk8, pv8);
        plan.num_slots =
            plan_slot_count(num_threads, use_thread_slots, std::max(tasks, B * H * NQ));
        size_t bytes =
            static_cast<size_t>(plan.num_slots) * static_cast<size_t>(per_task) * sizeof(T);
        if (splits > 1) {
            bytes = align_up_bytes(bytes, 16) +
                    static_cast<size_t>(tasks) * static_cast<size_t>(hb) *
                        static_cast<size_t>(Br) * static_cast<size_t>(Dv + 2) * sizeof(float) +
                    static_cast<size_t>(plan.num_slots) * static_cast<size_t>(Dv) * sizeof(float);
        }
        plan.workspace_size = bytes;
        return plan;
    }

    constexpr int mr_max = std::is_same_v<T, half> ? MR_MAX_F16 : MR_MAX_F32;
    constexpr int nr_max = nr_max_flt<T>();
    constexpr int kc = static_cast<int>(kc_for<T>());
    constexpr int esz = static_cast<int>(sizeof(T));

    int mc1, nc1;
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(Sk), mr_max, nr_max, kc, esz,
                       num_threads, mc1, nc1);
    int mc2, nc2;
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(Dv), mr_max, nr_max, kc, esz,
                       num_threads, mc2, nc2);
    plan.mc1 = mc1;
    plan.nc1 = nc1;
    plan.mc2 = mc2;
    plan.nc2 = nc2;

    // One packed-B slot for both GEMMs: stride from the longer k-block, panels
    // from the wider n-block. The ZA engine adds its packed query block and one
    // probability tile, both sized by the same helpers the kernel carves with.
    const int64_t scores_elems = Sq * Sk;
    const int64_t pack_elems = standard_b_pack_elems<T>(nc1, nc2, Dqk, Dv, Sk);
    const int64_t i8_elems =
        kv8 ? kv_i8_panel_elems<T>(kc_pack_bound<T>(std::max(Dqk, Sk))) : 0;
    const int64_t sme_elems = standard_sme_q_elems<T>(Sq, Dqk) +
                              standard_sme_p_elems<T>(mc2, Sk);
    const int64_t per_head = scores_elems + pack_elems + i8_elems + sme_elems;

    plan.num_slots = plan_slot_count(num_threads, use_thread_slots, B * H);
    plan.workspace_size =
        static_cast<size_t>(plan.num_slots) * static_cast<size_t>(per_head) * sizeof(T);
    return plan;
}

} // anonymous namespace

AttentionPlan get_attention_plan(const AttentionConfig& cfg, std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> /*outputs*/, int num_threads,
                                 bool use_thread_slots)
{
    NNOPS_ASSERT(inputs.size() >= 2);

    const DataType dt = inputs[0].dtype;

    // The K/V cache may be int8 while the query stays fp; anything else must
    // share the query's dtype, and a mismatch returns a zeroed plan.
    const bool kv8 = inputs[1].dtype == DataType::s8;
    if (inputs[1].dtype != dt && !(kv8 && (dt == DataType::f32 || dt == DataType::f16))) {
        return AttentionPlan{};
    }
    if (!cfg.v_is_k && inputs.size() > 2 && inputs[2].dtype != inputs[1].dtype) {
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

} // namespace nnops::backend::cpu
