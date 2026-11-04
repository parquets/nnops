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
    bool merged;
};

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
    return s;
}

/// Offset of the value head inside a merged row: a latent K row is Dqk wide
/// per head even though only Dv of those columns are values.
inline int64_t merged_v_head_stride(const AttentionConfig& cfg, int64_t Dqk, int64_t Dv)
{
    return cfg.v_is_k ? Dqk : Dv;
}

// ---- causal row limit ----------------------------------------------------

/// Exclusive end of query row @p i's valid key range under the config:
/// Sk when not causal, else the bottom-right-aligned prefix end.
inline int64_t causal_limit(const AttentionConfig& cfg, int64_t i,
                            int64_t Sq, int64_t Sk)
{
    if (!cfg.causal) { return Sk; }
    const int64_t lim = i + (Sk - Sq) + 1;
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
    const bool has_mask = inputs.size() > (cfg.v_is_k ? 2u : 3u);

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Sk = hs.Sk, Dqk = hs.Dqk, Dv = hs.Dv;
    const bool merged = hs.merged;

    const float scale = (cfg.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* mask_ptr = has_mask ? inputs[cfg.v_is_k ? 2 : 3].ptr<T>() : nullptr;
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
    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < NG;

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

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : idx;
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

inline bool should_use_flash_attention(int64_t Sq, int64_t Sk)
{
    return Sq * Sk >= kFlashMinScores;
}

inline void resolve_flash_tile_sizes(size_t elem_size, int64_t Sq, int64_t Sk,
                                     int64_t Dqk, int64_t Dv, int& Br, int& Bc)
{
    const size_t l2 = simd::CpuFeatures::get().l2_cache_size();
    const size_t l2_eff = l2 > 0 ? l2 : (1024u * 1024u);  // 1 MiB fallback
    const int64_t head_pairs = Dqk + Dv;  // qk_head_size + v_head_size
    Bc = static_cast<int>(l2_eff / (elem_size * 4 * static_cast<size_t>(head_pairs)));
    if (Bc < 1) { Bc = 1; }
    Br = static_cast<int>(std::min<int64_t>(Bc, head_pairs));
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
    const bool has_mask = inputs.size() > (cfg.v_is_k ? 2u : 3u);

    const HeadShape hs = resolve_head_shape(cfg, inputs);
    const int64_t B = hs.B, H = hs.H, Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq, Sk = hs.Sk, Dqk = hs.Dqk, Dv = hs.Dv;
    const bool merged = hs.merged;

    const float scale = (cfg.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(Dqk)))
        : cfg.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* mask_ptr = has_mask ? inputs[cfg.v_is_k ? 2 : 3].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
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
    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < NG;

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
            k_head = k_ptr + b * Sk * k_rs + hkv * Dqk;
            v_head = v_ptr + b * hs.Sv * v_rs + hkv * v_head_stride;
            o_head = out_ptr + b * Sq * o_rs + h * Dv;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * Hkv + hkv) * Sk * k_rs;
            v_head = v_ptr + (b * Hkv + hkv) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : idx;
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
        // o_acc is written by the first KV block (zero_mode) — no pre-zeroing
        // pass. The exp_diff rescale may touch it first on j0 == 0; that
        // garbage is fully overwritten by the first MMA.
        for (int64_t j0 = 0; j0 < Sk; j0 += Bc) {
            const int actual_bc = static_cast<int>(std::min<int64_t>(Bc, Sk - j0));

            // ---- S = Q[i0:i0+Br] @ K[j0:j0+Bc]^T * scale ----------
            // k0 == 0 overwrites (zero_mode) — each KV block owns its S tile.
            for (int64_t k0 = 0; k0 < Dqk; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Dqk - k0));
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                    / static_cast<int>(sizeof(T)));
                tile_pack_rhs(true, actual_bc, actual_kc,
                                      pack_b, ldd_b,
                                      k_head + j0 * k_rs + k0, static_cast<int>(k_rs), scale);
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
                                      v_head + (j0 + k0) * v_rs, static_cast<int>(v_rs),
                                      1.0f);
                tile_mma_direct(actual_br, static_cast<int>(Dv), actual_kc,
                                        o_acc, static_cast<int>(Dv),
                                        s + k0, actual_bc,
                                        pack_b, /*ldb=*/-1, /*stage=*/nullptr, kNegInf, kPosInf, j0 == 0 && k0 == 0);
            }
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

}  // anonymous namespace

// =========================================================================
//  Plan — single source of truth for tile sizes + workspace layout
// =========================================================================

namespace {

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
    int64_t B, Sq, Sk, Dqk;
    if (merged) {
        B  = q.dims[0];
        Sq = q.dims[1];
        Sk = k.dims[1];
        Dqk = q.dims[2] / H;
    } else {
        B  = q.dims[0];
        Sq = q.dims[2];
        Sk = k.dims[2];
        Dqk = q.dims[3];
    }
    const int64_t Dv = cfg.v_is_k ? cfg.v_head_dim : Dqk;

    plan.use_flash = should_use_flash_attention(Sq, Sk);

    // Flash path — sized with the same Br/Bc the kernel uses. Slots are
    // per worker thread when thread ids are reported, else per query block.
    if (plan.use_flash) {
        int Br, Bc;
        resolve_flash_tile_sizes(sizeof(T), Sq, Sk, Dqk, Dv, Br, Bc);
        plan.Br = Br;
        plan.Bc = Bc;
        const int64_t per_task = flash_per_task_elems<T>(Br, Bc, Dqk, Dv);
        const int64_t NQ = (Sq + Br - 1) / Br;
        const int64_t nt  = std::max<int64_t>(num_threads, 1);
        plan.num_slots = (use_thread_slots && nt < B * H * NQ) ? nt : B * H * NQ;
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

    const int64_t nt = std::max<int64_t>(num_threads, 1);
    plan.num_slots = (use_thread_slots && nt < B * H) ? nt : B * H;
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
        if (plan.use_flash) {
            attention_flash_impl<half>(cfg, output, inputs, ctx, plan, scratch.as<half>());
        } else {
            attention_impl<half>(cfg, output, inputs, ctx, plan, scratch.as<half>());
        }
    } else {
        if (plan.use_flash) {
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
    cfg.v_is_k = true;
    return attention_cpu(cfg, output, inputs, ctx, workspace);
}

}  // namespace nnops::backend::cpu
