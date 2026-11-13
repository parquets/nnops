/// @file attention_standard.cpp
/// @brief The materialized-scores route: scores = Q·Kᵀ·scale, softmax each
///        row in place, output = attn·V.
///
/// This is the small-shape route (Sq*Sk < kFlashMinScores). It is the only one
/// that builds the whole per-head score matrix, which is why it needs the
/// largest scratch slot and why very long prefills are sent to flash instead.
///
/// One task is one (batch, query head). Both GEMMs drive the packed-B route:
/// the first packs K with the scale folded in, the second packs V. Both run
/// their first k-block with zero_mode (add_to=false) — nothing is pre-zeroed,
/// and that block skips the C read.
///
/// The mask folds into the softmax 3-pass (max → exp+sum → normalize), so
/// scores is never rewritten before normalization — one full-buffer pass
/// cheaper than an explicit `scores += mask`. A causal row only softmaxes its
/// valid prefix; the masked tail is zeroed so GEMM2 reads nothing from it.

#include "attention_internal.h"

namespace nnops::backend::cpu {

namespace {

template <class T>
void attention_standard_impl(const AttentionConfig& cfg,
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
    const int32_t* seqlens = has_seqlens_input(cfg, inputs)
        ? inputs[seqlens_slot(cfg)].ptr<int32_t>() : nullptr;

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

        // The batch's valid key run: its prefix of the [past | new] rows.
        const int64_t run = run_length(seqlens, b, Sk);

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
            const int64_t n = causal_limit(cfg, i, Sq, run);
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

}  // anonymous namespace

void attention_standard_cpu(const AttentionConfig& cfg,
                            TensorView& output,
                            std::span<const TensorView> inputs,
                            const ComputeContext& ctx,
                            const AttentionPlan& plan,
                            void* scratch)
{
    NNOPS_ASSERT(inputs[0].data_type() == DataType::f16 ||
                 inputs[0].data_type() == DataType::f32 ||
                 inputs[0].data_type() == DataType::bf16);
    if (inputs[0].data_type() == DataType::f16) {
        attention_standard_impl<half>(cfg, output, inputs, ctx, plan, scratch);
#ifdef NNOPS_ARCH_AARCH64
    } else if (inputs[0].data_type() == DataType::bf16) {
        attention_standard_impl<bfloat16>(cfg, output, inputs, ctx, plan, scratch);
#endif
    } else {
        attention_standard_impl<float>(cfg, output, inputs, ctx, plan, scratch);
    }
}

}  // namespace nnops::backend::cpu
