#pragma once
/// @file attention_common.cuh
/// @brief The contract the two CUDA attention kernels share — the head-geometry
///        description, the three storage orders' strides, the float / s8 K/V
///        accessors and the output-gate activation.
///
/// Split out of attention.cu (the layout follows cuda/conv's conv2d_common.cuh
/// and cuda/matmul's gemm_common.cuh) so attention_kernel.cuh (the tiled
/// prefill kernel) and attention_decode.cuh (the one-row decode kernel) read the
/// same description and the same mask semantics.

#include "backend/common/attention_config.hpp"
#include "backend/cuda/cuda_common.cuh"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <type_traits>

namespace nnops::backend::cuda::attn {

constexpr int kBr = 16;       ///< query rows one block owns
constexpr int kThreads = 256; ///< 16 rows × 16 lane groups
/// The all-masked sentinel. A finite -FLT_MAX rather than -inf so the softmax
/// can test it by equality: exp(s - m) underflows to zero either way, and no
/// -inf - -inf NaN can arise.
constexpr float kNegInf = -FLT_MAX;

/// Row stride for a shared tile whose rows a warp walks in parallel. An odd
/// stride is coprime with the 32 banks, so consecutive rows land on distinct
/// banks; an even one (a multiple of 32 in particular) puts every row on the
/// same bank and serialises a sixteen-row access.
__host__ __device__ inline int64_t pad_stride(int64_t n) { return n + ((n & 1) == 0 ? 1 : 0); }

// ============================================================
// Layout: the same three orders the CPU core addresses
// ============================================================

enum class HeadLayout { Merged, BHSD, BSHD };

inline int64_t head_seq_dim(HeadLayout l) { return l == HeadLayout::BHSD ? 2 : 1; }

inline HeadLayout q_layout_of(const AttentionConfig& cfg, const TensorView& q)
{
    if (q.rank() == 3) {
        return HeadLayout::Merged;
    }
    return cfg.bshd ? HeadLayout::BSHD : HeadLayout::BHSD;
}

inline HeadLayout kv_layout_of(const AttentionConfig& cfg, const TensorView& k)
{
    if (k.rank() == 3) {
        return HeadLayout::Merged;
    }
    return cfg.kv_bshd ? HeadLayout::BSHD : HeadLayout::BHSD;
}

/// One sequence position forward within one head. Merged and BNSH rows are the
/// head's own row, so the row pitch; BSHD steps a whole head block.
inline int64_t seq_step_of(const TensorView& v, HeadLayout l)
{
    return l == HeadLayout::BSHD ? v.stride_elems(1) : v.row_stride_elems();
}

/// One head forward. A merged row packs the heads, so the step is the head
/// width; the rank-4 forms carry their own axis.
inline int64_t head_step_of(const TensorView& v, HeadLayout l, int64_t d)
{
    return l == HeadLayout::Merged ? d
                                   : (l == HeadLayout::BHSD ? v.stride_elems(1) : v.stride_elems(2));
}

inline bool slot_present(const AttentionConfig& cfg, int slot, std::span<const TensorView> in)
{
    return slot >= 0 && in.size() > static_cast<size_t>(slot) && !in[static_cast<size_t>(slot)].is_empty();
}

// ============================================================
// Kernel parameters
// ============================================================

template <class T, bool KV8>
struct AttnParams {
    using KVT = std::conditional_t<KV8, int8_t, T>;
    static constexpr bool kv8 = KV8;

    // ---- geometry ----
    int64_t B = 0, H = 0, Hkv = 0, Sq = 0, Dqk = 0, Dv = 0;
    int64_t Sk = 0;      ///< run capacity: past rows + the new/current rows
    int64_t Sk_past = 0; ///< rows of the past segment (0 without one)
    int64_t group = 1;   ///< H / H_kv
    int head_batch = 1;  ///< decode: query heads one block serves (1..group)
    float scale = 0.0f;
    int bc = 32;         ///< keys per KV tile (32, 16 or 8)

    // ---- element strides (batch, head, sequence); d is contiguous ----
    int64_t q_b = 0, q_h = 0, q_s = 0;
    int64_t k_b = 0, k_h = 0, k_s = 0;
    int64_t v_b = 0, v_h = 0, v_s = 0;
    int64_t pk_b = 0, pk_h = 0, pk_s = 0;
    int64_t pv_b = 0, pv_h = 0, pv_s = 0;
    int64_t o_b = 0, o_h = 0, o_s = 0;
    int64_t g_b = 0, g_h = 0, g_s = 0;

    // ---- mask ----
    int64_t bias_b = 0, bias_h = 0, bias_row = 0;
    int64_t keep_b = 0, keep_row = 0;
    float filter = -10000.0f;

    // ---- metadata ----
    const int32_t* run_ptr = nullptr;
    bool run_plus_one = false;
    const int64_t* pos_ids = nullptr;
    int64_t position_offset = -1;
    const float* sink = nullptr;
    const float* kscale = nullptr;
    const float* vscale = nullptr;
    bool scale_per_token = false;

    // ---- pointers ----
    const T* q = nullptr;
    const KVT* k = nullptr;
    const KVT* v = nullptr;
    const KVT* pk = nullptr;
    const KVT* pv = nullptr;
    const T* bias = nullptr;
    const int32_t* keep = nullptr;
    const T* gate = nullptr;
    T* o = nullptr;

    // ---- flags ----
    bool causal = false;
    bool v_is_k = false;
    bool has_bias = false;
    bool has_keep = false;
    bool has_gate = false;
    bool softcap_on = false;
    float softcap = 0.0f;
    int64_t window = 0;
    int64_t sink_tokens = 0;
    int gate_act = 0; ///< EpilogueActivateType
};

// ============================================================
// Device helpers
// ============================================================

template <class P> __device__ __forceinline__ float kv_elem(const typename P::KVT* p)
{
    if constexpr (P::kv8) {
        return static_cast<float>(*p);
    } else {
        return s_load(p);
    }
}

/// Base of logical run row `j` (past segment first), or null past the run. The
/// head width is contiguous from there, so a caller that wants a whole row --
/// the decode kernel's register dot -- reads `kv_elem(row + d)` directly.
template <class P>
__device__ __forceinline__ const typename P::KVT* k_row(const P& p, int64_t b, int64_t hkv,
                                                        int64_t j)
{
    if (j < 0 || j >= p.Sk) {
        return nullptr;
    }
    if (j < p.Sk_past) {
        return p.pk + b * p.pk_b + hkv * p.pk_h + j * p.pk_s;
    }
    return p.k + b * p.k_b + hkv * p.k_h + (j - p.Sk_past) * p.k_s;
}

/// The same row of the value run. The absorbed/latent form takes the value from
/// the key row's leading columns.
template <class P>
__device__ __forceinline__ const typename P::KVT* v_row(const P& p, int64_t b, int64_t hkv,
                                                        int64_t j)
{
    if (p.v_is_k) {
        return k_row(p, b, hkv, j);
    }
    if (j < 0 || j >= p.Sk) {
        return nullptr;
    }
    if (j < p.Sk_past) {
        return p.pv + b * p.pv_b + hkv * p.pv_h + j * p.pv_s;
    }
    return p.v + b * p.v_b + hkv * p.v_h + (j - p.Sk_past) * p.v_s;
}

/// Key column `d` of logical run row `j`. Out-of-run rows (a tile's tail past
/// the capacity) read zero; they are masked anyway.
template <class P>
__device__ __forceinline__ float read_k(const P& p, int64_t b, int64_t hkv, int64_t j, int64_t d)
{
    const typename P::KVT* r = k_row(p, b, hkv, j);
    return r != nullptr ? kv_elem<P>(r + d) : 0.0f;
}

/// Value column `d` for the same run row.
template <class P>
__device__ __forceinline__ float read_v(const P& p, int64_t b, int64_t hkv, int64_t j, int64_t d)
{
    const typename P::KVT* r = v_row(p, b, hkv, j);
    return r != nullptr ? kv_elem<P>(r + d) : 0.0f;
}

/// The cache's dequantization scale of row `j`: one value per tensor or one per
/// (batch, kv head, row).
__device__ __forceinline__ float kv_scale_at(const float* ptr, bool per_token, int64_t b,
                                             int64_t hkv, int64_t j, int64_t Hkv, int64_t Sk)
{
    if (ptr == nullptr) {
        return 1.0f;
    }
    return per_token ? ptr[(b * Hkv + hkv) * Sk + j] : ptr[0];
}

__device__ __forceinline__ float gate_act_value(int act, float g)
{
    switch (act) {
    case int(EpilogueActivateType::Silu):
        return g / (1.0f + __expf(-g));
    case int(EpilogueActivateType::Gelu):
        return 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g)));
    case int(EpilogueActivateType::Relu):
        return g > 0.0f ? g : 0.0f;
    case int(EpilogueActivateType::Sigmoid):
        return 1.0f / (1.0f + __expf(-g));
    case int(EpilogueActivateType::Tanh):
        return tanhf(g);
    default:
        return g;
    }
}

} // namespace nnops::backend::cuda::attn
