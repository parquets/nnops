/// @file attention.cpp
/// @brief Tiled multi-head scaled dot-product attention — fused QK^T + softmax + AV.
///
/// Two routes, picked by the per-head scores area:
///   - standard (Sq*Sk < kFlashMinScores): materialize scores = Q·K^T·scale,
///     softmax each row in place, then output = attn·V.
///   - flash (Sq*Sk >= kFlashMinScores): FlashAttention-1 online softmax,
///     streaming KV in Br×Bc tiles so scores is never materialized. The task
///     grid is batch × head × query-block, so long-sequence prefill keeps its
///     parallelism even when batch × head is small.
///
/// The scale is fused into the GEMM1 RHS pack. Both GEMMs only accumulate, so
/// the first k-block of each output tile runs with zero_mode (add_to=false):
/// nothing is pre-zeroed, and that block skips the C read.
///
/// Mask handling differs by route: the standard path folds it into the
/// mask+softmax kernel (one pass fewer than `scores += mask`), the flash path
/// adds it into the S tile before the online-softmax update.
///
/// Scratch sizing counts uniform nr_max-wide panels, not a width cascade: the
/// widest n-block in a loop bounds the count (a greedy per-width decomposition
/// is not monotonic in n and undersized the slot). The packed-B stride takes
/// min(Kc, k) of the longer GEMM, since a head dim below Kc would otherwise
/// over-allocate by up to 2x.
///
/// See attention.h for the per-(batch, head) algorithm, the layout contract
/// and the scratch layout.

#include "attention.h"
#include "backend/cpu/matmul/matmul_helper.h"                  // tile_mma_direct / tile_pack_rhs + panel constants
#include "backend/cpu/simd_kernel/simd_softmax.hpp"     // softmax + FlashAttention row kernels
#include "backend/cpu/common/memory_pool.hpp"           // internal scratch-memory pool
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
    int64_t B, H, Sq, Sk, Sv, D;
    bool merged;
};

inline HeadShape resolve_head_shape(const AttentionAttributes& attrs,
                                    std::span<const TensorView> inputs)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];

    HeadShape s;
    s.B = Q.shape(0);
    s.H = attrs.num_heads;
    s.merged = (Q.rank() == 3);
    if (s.merged) {
        s.Sq = Q.shape(1);
        s.Sk = K.shape(1);
        s.Sv = V.shape(1);
        s.D  = Q.shape(2) / s.H;
    } else {
        s.Sq = Q.shape(2);
        s.Sk = K.shape(2);
        s.Sv = V.shape(2);
        s.D  = Q.shape(3);
    }
    return s;
}

// ---- per-(batch,head) tiled kernel ---------------------------------------

template <class T>
void attention_impl(const AttentionAttributes& attrs,
                    TensorView& output,
                    std::span<const TensorView> inputs,
                    const ComputeContext& ctx,
                    const AttentionPlan& plan,
                    void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];
    const bool has_mask = inputs.size() > 3;

    const HeadShape hs = resolve_head_shape(attrs, inputs);
    const int64_t B = hs.B, H = hs.H, Sq = hs.Sq, Sk = hs.Sk, D = hs.D;
    const bool merged = hs.merged;

    const float scale = (attrs.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(D)))
        : attrs.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* mask_ptr = has_mask ? inputs[3].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t o_rs = output.row_stride_elems();

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int64_t kc = kc_for<T>();

    const int mc1 = plan.mc1;
    const int nc1 = plan.nc1;
    const int mc2 = plan.mc2;
    const int nc2 = plan.nc2;

    // Scratch bounds: longest k-block and widest n-block either GEMM sees.
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(
            kc_pack_bound<T>(std::max(D, Sk))) * static_cast<int>(sizeof(T)))
        / static_cast<int>(sizeof(T)));
    const int np_full = std::max(panels_for_pack<T>(nc1),
                                 panels_for_pack<T>(std::min<int64_t>(D, nc2)));

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

        const T* q_head;
        const T* k_head;
        const T* v_head;
        T* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * D;
            k_head = k_ptr + b * Sk * k_rs + h * D;
            v_head = v_ptr + b * hs.Sv * v_rs + h * D;
            o_head = out_ptr + b * Sq * o_rs + h * D;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * H + h) * Sk * k_rs;
            v_head = v_ptr + (b * H + h) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : idx;
        T* scores = ws_base + slot * per_head;
        T* pack_b = scores + scores_elems;

        // ---- GEMM1: scores = Q @ K^T * scale --------------------------
        // k0 == 0 runs zero_mode (overwrite) — no pre-zeroing pass.
        for (int64_t k0 = 0; k0 < D; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, D - k0));
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
                                            pack_b,
                                            kNegInf, kPosInf, k0 == 0);
                }
            }
        }

        // ---- mask + softmax over the last dim, per row, in-place ------
        // The mask folds into the softmax 3-pass (max → exp+sum → normalize),
        // so scores is never rewritten before normalization — one full-buffer
        // pass cheaper than an explicit `scores += mask`.
        for (int64_t i = 0; i < Sq; ++i) {
            if (mask_ptr) {
                kernel::mask_softmax_process_standard_row<T>(
                    scores + i * Sk, mask_ptr + i * Sk, scores + i * Sk, Sk,
                    /*log_softmax=*/false, 1.0f);
            } else {
                kernel::softmax_process_standard_row<T>(
                    scores + i * Sk, scores + i * Sk, Sk, /*log_softmax=*/false, 1.0f);
            }
        }

        // ---- GEMM2: output = attn @ V ---------------------------------
        for (int64_t k0 = 0; k0 < Sk; k0 += kc) {
            const int actual_kc = static_cast<int>(std::min<int64_t>(kc, Sk - k0));
            const int ldd_b = static_cast<int>(
                mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                / static_cast<int>(sizeof(T)));
            for (int64_t n = 0; n < D; n += nc2) {
                const int actual_nc = static_cast<int>(std::min<int64_t>(nc2, D - n));
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
                                            pack_b, kNegInf, kPosInf, k0 == 0);
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
// cache. Br / Bc follow the MLAS L2 heuristic: Bc = L2 / (sizeof(T)·4·2D) and
// Br = min(Bc, 2D), then both capped to Sk / Sq.

constexpr int64_t kFlashMinScores = 1 << 16;  // Sq*Sk threshold (≈ 256 KiB f32)

inline bool should_use_flash_attention(int64_t Sq, int64_t Sk)
{
    return Sq * Sk >= kFlashMinScores;
}

template <class T>
inline void resolve_flash_tile_sizes(int64_t Sq, int64_t Sk, int64_t D, int& Br, int& Bc)
{
    const size_t l2 = simd::CpuFeatures::get().l2_cache_size();
    const size_t l2_eff = l2 > 0 ? l2 : (1024u * 1024u);  // 1 MiB fallback
    const int64_t head_pairs = D + D;  // qk_head_size + v_head_size (both == D here)
    Bc = static_cast<int>(l2_eff / (sizeof(T) * 4 * head_pairs));
    if (Bc < 1) { Bc = 1; }
    Br = static_cast<int>(std::min<int64_t>(Bc, head_pairs));
    Bc = static_cast<int>(std::min<int64_t>(Bc, Sk));
    Br = static_cast<int>(std::min<int64_t>(Br, Sq));
    if (Br < 1) { Br = 1; }
    if (Bc < 1) { Bc = 1; }
}

// Per-task scratch for one Br-row query block, matching the layout inside
// attention_flash_impl: [m:Br][l:Br][s:Br*Bc][o_acc:Br*D][pack_b]. The
// packed-B slot serves GEMM1 (K = D, N <= Bc) and GEMM2 (K <= Bc, N = D).
template <class T>
inline int64_t flash_per_task_elems(int Br, int Bc, int64_t D)
{
    constexpr int nr_max = nr_max_flt<T>();
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(
            kc_pack_bound<T>(std::max<int64_t>(D, Bc))) * static_cast<int>(sizeof(T)))
        / static_cast<int>(sizeof(T)));
    const int np_full = std::max(panels_for_pack<T>(Bc), panels_for_pack<T>(D));
    const int pack_elems = np_full * ldd_b_full;
    return static_cast<int64_t>(Br) * 2
         + static_cast<int64_t>(Br) * Bc
         + static_cast<int64_t>(Br) * D
         + pack_elems;
}

// ---- tiled FlashAttention kernel -----------------------------------------
//
// One task = one query block (Br rows) of one (batch, head): it streams KV in
// Bc-sized blocks, carrying a running max/sum and an output accumulator, so
// the Sq×Sk scores matrix never exists. Reuses the packed-B route above.

template <class T>
void attention_flash_impl(const AttentionAttributes& attrs,
                          TensorView& output,
                          std::span<const TensorView> inputs,
                          const ComputeContext& ctx,
                          const AttentionPlan& plan,
                          void* workspace)
{
    const auto& Q = inputs[0];
    const auto& K = inputs[1];
    const auto& V = inputs[2];
    const bool has_mask = inputs.size() > 3;

    const HeadShape hs = resolve_head_shape(attrs, inputs);
    const int64_t B = hs.B, H = hs.H, Sq = hs.Sq, Sk = hs.Sk, D = hs.D;
    const bool merged = hs.merged;

    const float scale = (attrs.scale == 0.0f)
        ? (1.0f / std::sqrt(static_cast<float>(D)))
        : attrs.scale;

    const T* q_ptr = Q.ptr<T>();
    const T* k_ptr = K.ptr<T>();
    const T* v_ptr = V.ptr<T>();
    const T* mask_ptr = has_mask ? inputs[3].ptr<T>() : nullptr;
    T* out_ptr = output.ptr<T>();

    const int64_t q_rs = Q.row_stride_elems();
    const int64_t k_rs = K.row_stride_elems();
    const int64_t v_rs = V.row_stride_elems();
    const int64_t o_rs = output.row_stride_elems();

    constexpr int nr_max = nr_max_flt<T>();
    constexpr int64_t kc = kc_for<T>();
    constexpr int L = simd::simd_lane_for<T>;

    const int Br = plan.Br;
    const int Bc = plan.Bc;
    const int64_t per_task = flash_per_task_elems<T>(Br, Bc, D);

    const int64_t NQ = (Sq + Br - 1) / Br;  // number of Br-row query blocks
    const int64_t NG = B * H * NQ;
    T* ws_base = static_cast<T*>(workspace);
    const int nslots = ctx.cpu.thread_slot_count();
    const bool per_thread = nslots > 0 && nslots < NG;

    const auto run = [&](int64_t idx) {
        const int64_t qi = idx % NQ;
        const int64_t b  = (idx / NQ) / H;
        const int64_t h  = (idx / NQ) % H;
        const int64_t i0 = qi * Br;
        const int actual_br = static_cast<int>(std::min<int64_t>(Br, Sq - i0));

        const T* q_head;
        const T* k_head;
        const T* v_head;
        T* o_head;
        if (merged) {
            q_head = q_ptr + b * Sq * q_rs + h * D;
            k_head = k_ptr + b * Sk * k_rs + h * D;
            v_head = v_ptr + b * hs.Sv * v_rs + h * D;
            o_head = out_ptr + b * Sq * o_rs + h * D;
        } else {
            q_head = q_ptr + (b * H + h) * Sq * q_rs;
            k_head = k_ptr + (b * H + h) * Sk * k_rs;
            v_head = v_ptr + (b * H + h) * hs.Sv * v_rs;
            o_head = out_ptr + (b * H + h) * Sq * o_rs;
        }

        const int64_t slot = per_thread ? static_cast<int64_t>(ctx.cpu.current_thread_id()) : idx;
        T* m      = ws_base + slot * per_task;
        T* l      = m + Br;
        T* s      = l + Br;
        T* o_acc  = s + static_cast<int64_t>(Br) * Bc;
        T* pack_b = o_acc + static_cast<int64_t>(Br) * D;

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
            for (int64_t k0 = 0; k0 < D; k0 += kc) {
                const int actual_kc = static_cast<int>(std::min<int64_t>(kc, D - k0));
                const int ldd_b = static_cast<int>(
                    mma_pack_align(nr_max * actual_kc * static_cast<int>(sizeof(T)))
                    / static_cast<int>(sizeof(T)));
                tile_pack_rhs(true, actual_bc, actual_kc,
                                      pack_b, ldd_b,
                                      k_head + j0 * k_rs + k0, static_cast<int>(k_rs), scale);
                tile_mma_direct(actual_br, actual_bc, actual_kc,
                                        s, actual_bc,
                                        q_head + i0 * q_rs + k0, static_cast<int>(q_rs),
                                        pack_b, kNegInf, kPosInf, k0 == 0);
            }

            // ---- additive mask (flat [Sq, Sk]: no batch or head index) ----
            if (mask_ptr) {
                for (int i = 0; i < actual_br; ++i) {
                    const T* mrow = mask_ptr + (i0 + i) * Sk + j0;
                    T* srow = s + i * actual_bc;
                    int j = 0;
                    for (; j + L <= actual_bc; j += L) {
                        simd::v_store(srow + j, simd::v_add(simd::v_load(srow + j), simd::v_load(mrow + j)));
                    }
                    for (; j < actual_bc; ++j) {
                        simd::s_store(&srow[j], simd::s_load(&srow[j]) + simd::s_load(&mrow[j]));
                    }
                }
            }

            // ---- online-softmax update + accumulate into o_acc -----
            for (int i = 0; i < actual_br; ++i) {
                T* srow = s + i * actual_bc;
                const float rowmax = kernel::softmax_row_max<T>(srow, actual_bc);
                const float old_m = simd::s_load(&m[i]);
                const float new_m = std::max(old_m, rowmax);
                const float exp_diff = std::exp(old_m - new_m);  // 0 on the first KV block
                simd::s_store(&m[i], new_m);
                const float rowsum = kernel::softmax_row_exp_sum<T>(srow, srow, actual_bc, new_m, 1.0f);
                simd::s_store(&l[i], exp_diff * simd::s_load(&l[i]) + rowsum);

                if (exp_diff != 1.0f) {
                    T* orow = o_acc + i * D;
                    auto v_ed = simd::v_set1(orow, exp_diff);
                    int64_t d = 0;
                    for (; d + L <= D; d += L) {
                        simd::v_store(orow + d, simd::v_mul(simd::v_load(orow + d), v_ed));
                    }
                    for (; d < D; ++d) {
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
                // V rows [j0+k0, +actual_kc) x D, packed once for the whole
                // Br-row block.
                tile_pack_rhs(false, static_cast<int>(D), actual_kc,
                                      pack_b, ldd_b,
                                      v_head + (j0 + k0) * v_rs, static_cast<int>(v_rs),
                                      1.0f);
                tile_mma_direct(actual_br, static_cast<int>(D), actual_kc,
                                        o_acc, static_cast<int>(D),
                                        s + k0, actual_bc,
                                        pack_b, kNegInf, kPosInf, j0 == 0 && k0 == 0);
            }
        }

        // ---- final normalize: output = o_acc / l -------------------
        for (int i = 0; i < actual_br; ++i) {
            const float inv_l = 1.0f / simd::s_load(&l[i]);
            const T* src = o_acc + i * D;
            T* dst = o_head + (i0 + i) * o_rs;
            auto v_inv = simd::v_set1(src, inv_l);
            int64_t d = 0;
            for (; d + L <= D; d += L) {
                simd::v_store(dst + d, simd::v_mul(simd::v_load(src + d), v_inv));
            }
            for (; d < D; ++d) {
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
AttentionPlan attention_plan_for(const AttentionAttributes& attrs,
                                 std::span<const TensorDesc> inputs,
                                 int num_threads,
                                 bool use_thread_slots)
{
    const auto& q = inputs[0];
    const auto& k = inputs[1];

    AttentionPlan plan;

    const int64_t H = attrs.num_heads;
    const bool merged = (q.rank == 3);
    int64_t B, Sq, Sk, D;
    if (merged) {
        B  = q.dims[0];
        Sq = q.dims[1];
        Sk = k.dims[1];
        D  = q.dims[2] / H;
    } else {
        B  = q.dims[0];
        Sq = q.dims[2];
        Sk = k.dims[2];
        D  = q.dims[3];
    }

    plan.use_flash = should_use_flash_attention(Sq, Sk);

    // Flash path — sized with the same Br/Bc the kernel uses. Slots are
    // per worker thread when thread ids are reported, else per query block.
    if (plan.use_flash) {
        int Br, Bc;
        resolve_flash_tile_sizes<T>(Sq, Sk, D, Br, Bc);
        plan.Br = Br;
        plan.Bc = Bc;
        const int64_t per_task = flash_per_task_elems<T>(Br, Bc, D);
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
    resolve_tile_sizes(static_cast<int>(Sq), static_cast<int>(D),
                       mr_max, nr_max, kc, esz, num_threads, mc2, nc2);
    plan.mc1 = mc1; plan.nc1 = nc1;
    plan.mc2 = mc2; plan.nc2 = nc2;

    // One packed-B slot for both GEMMs: stride from the longer k-block, panels
    // from the wider n-block.
    const int ldd_b_full = static_cast<int>(
        mma_pack_align(nr_max * static_cast<int>(
            kc_pack_bound<T>(std::max(D, Sk))) * esz) / esz);
    const int np_full = std::max(panels_for_pack<T>(nc1),
                                 panels_for_pack<T>(std::min<int64_t>(D, nc2)));

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

AttentionPlan get_attention_plan(const AttentionAttributes& attrs,
                                 std::span<const TensorDesc> inputs,
                                 std::span<const TensorDesc> /*outputs*/,
                                 int num_threads,
                                 bool use_thread_slots)
{
    NNOPS_ASSERT(inputs.size() >= 3);
    NNOPS_ASSERT(inputs.size() <= 4);

    const DataType dt = inputs[0].dtype;

    // f32 and f16 take the tiled path; any other dtype, or a Q/K/V mismatch,
    // returns a zeroed plan.
    const bool types_match = (inputs[1].dtype == dt) && (inputs[2].dtype == dt);
    if (!types_match) { return AttentionPlan{}; }
    if (dt == DataType::f16) {
        return attention_plan_for<half>(attrs, inputs, num_threads, use_thread_slots);
    }
    if (dt == DataType::f32) {
        return attention_plan_for<float>(attrs, inputs, num_threads, use_thread_slots);
    }
    return AttentionPlan{};
}

// =========================================================================
//  Public API
// =========================================================================

Status attention_cpu(const AttentionAttributes& attrs,
                      TensorView& output,
                      std::span<const TensorView> inputs,
                      const ComputeContext& ctx,
                      void* /*workspace*/)
{
    const DataType dt = inputs[0].data_type();

    const bool dtypes_ok = (dt == DataType::f32 || dt == DataType::f16) &&
                           inputs[1].data_type() == dt &&
                           inputs[2].data_type() == dt &&
                           output.data_type() == dt &&
                           (inputs.size() <= 3 || inputs[3].data_type() == dt);

    if (dtypes_ok) {
        // One plan decides tiles and scratch, and the scratch is allocated once
        // here, outside the parallel dispatch — no per-task allocation.
        const TensorDesc descs[] = {inputs[0].desc(), inputs[1].desc(), inputs[2].desc()};
        const TensorDesc outs[]  = {output.desc()};
        const AttentionPlan plan = get_attention_plan(attrs, descs, outs,
                                                      ctx.cpu.thread_count(),
                                                      ctx.cpu.thread_slot_count() > 0);
        PoolPtr scratch(plan.workspace_size);

        if (dt == DataType::f16) {
            if (plan.use_flash) {
                attention_flash_impl<half>(attrs, output, inputs, ctx, plan, scratch.as<half>());
            } else {
                attention_impl<half>(attrs, output, inputs, ctx, plan, scratch.as<half>());
            }
        } else {
            if (plan.use_flash) {
                attention_flash_impl<float>(attrs, output, inputs, ctx, plan, scratch.as<float>());
            } else {
                attention_impl<float>(attrs, output, inputs, ctx, plan, scratch.as<float>());
            }
        }
        return Status::ok;
    }

    return Status::unsupported;
}

}  // namespace nnops::backend::cpu
