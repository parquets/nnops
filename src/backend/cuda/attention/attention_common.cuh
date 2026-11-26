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
#include "backend/cuda/matmul/ptx.h" // PTX_SM80_ENABLED gates the mma wrappers below
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
    /// The value run widened once into a scratch buffer (attention_i8's fp
    /// P·V): the same `[batch][kv head][row][d]` order, already scaled.
    const void* vprep = nullptr;
    /// The same run turned over for the integer P·V — `[batch][kv head][d][row]`
    /// s8, zero-padded to `vt_s` rows — so its B panel needs no in-tile
    /// transpose.
    const void* vt = nullptr;
    int64_t vt_s = 0; ///< row stride (keys) of `vt`
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
    /// The operator's dynamic integer route (cfg.quantize_q / cfg.quantize_pv):
    /// attention_i8.cuh runs Q·Kᵀ on the s8 atom when the first is set, and P·V
    /// on it when both are. The fp routes ignore them and dequantize on read.
    bool quantize_q = false;
    bool quantize_pv = false;
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

// ============================================================
// The tensor-core epilogue — shared by attention_f16.cuh and attention_i8.cuh
// ============================================================
//
// An output element is the same whatever produced it: value times the softmax
// denominator, optionally times the gate activation, stored in the query dtype.
// The two tiles differ in their product and their panels, never in this.
/// One mma into four fp32 accumulators, the accumulator as the C operand.
template <class T>
__device__ __forceinline__ void hmma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                     uint32_t b1)
{
#if PTX_SM80_ENABLED
    if constexpr (std::is_same_v<T, __half>) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                     "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
#endif
}

/// exp2, one MUFU. The softmax runs in log2 units — the scores are scaled by
/// `log2(e)` on the way in and the head sink converted on the way out — so the
/// exponential is this without the multiply `__expf` hides inside it.
__device__ __forceinline__ float ex2(float x)
{
    float r;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

/// Two floats as one packed 16-bit pair — the A fragment of a PV step.
template <class T> __device__ __forceinline__ uint32_t pack2(float lo, float hi)
{
    if constexpr (std::is_same_v<T, __half>) {
        const __half2 v = __floats2half2_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    } else {
        const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
}

/// The gate activation, with the choice a template argument: a constant `act`
/// folds the switch in `gate_act_value` away, so one arm is emitted instead of
/// five. That is not what makes the kernel small — the arms are still unrolled
/// once per stored element — it is what makes the *gated* instantiation cost
/// 4.7x the ungated one instead of dwarfing it.
template <int kAct> __device__ __forceinline__ float gate_value(float g)
{
    return gate_act_value(kAct, g);
}

/// The store pass. The four row pointers are formed once a lane instead of once
/// an element, and `kAct < 0` — the common case — is a compile-time form that
/// carries no activation at all.
template <class T, int NT, int kAct, class P>
__device__ __forceinline__ void store_output(const P& p, const float (&o)[NT][4], int64_t obase,
                                             int64_t gbase, int64_t i0, int c0, int row0,
                                             int row1, bool live0, bool live1, float scl0,
                                             float scl1, float norm0, float norm1)
{
    T* const dst0 = p.o + obase + (i0 + row0) * p.o_s;
    T* const dst1 = p.o + obase + (i0 + row1) * p.o_s;
    const T* const grow0 = p.gate + gbase + (i0 + row0) * p.g_s;
    const T* const grow1 = p.gate + gbase + (i0 + row1) * p.g_s;
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const int col = 8 * t + c0;
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            if (!(half == 0 ? live0 : live1)) {
                continue;
            }
            const float scl = half == 0 ? scl0 : scl1;
            const float norm = half == 0 ? norm0 : norm1;
            T* const dst = half == 0 ? dst0 : dst1;
            float v0 = (half == 0 ? o[t][0] : o[t][2]) * scl * norm;
            float v1 = (half == 0 ? o[t][1] : o[t][3]) * scl * norm;
            if constexpr (kAct >= 0) {
                const T* const grow = half == 0 ? grow0 : grow1;
                v0 *= gate_value<kAct>(s_load(grow + col));
                v1 *= gate_value<kAct>(s_load(grow + col + 1));
            }
            s_store(dst + col, v0);
            s_store(dst + col + 1, v1);
        }
    }
}

/// Pick the store pass's activation once for the block, so the element loop is
/// instantiated once an activation rather than once an element.
template <class T, int NT, class P>
__device__ __forceinline__ void store_output_dispatch(const P& p, const float (&o)[NT][4],
                                                      int64_t obase, int64_t gbase, int64_t i0,
                                                      int c0, int row0, int row1, bool live0,
                                                      bool live1, float scl0, float scl1,
                                                      float norm0, float norm1)
{
#define NNOPS_TC_STORE(ACT)                                                          \
    store_output<T, NT, (ACT)>(p, o, obase, gbase, i0, c0, row0, row1, live0, live1, \
                               scl0, scl1, norm0, norm1)
    if (!p.has_gate) {
        NNOPS_TC_STORE(-1);
        return;
    }
    switch (p.gate_act) {
    case int(EpilogueActivateType::Silu): NNOPS_TC_STORE(int(EpilogueActivateType::Silu)); return;
    case int(EpilogueActivateType::Gelu): NNOPS_TC_STORE(int(EpilogueActivateType::Gelu)); return;
    case int(EpilogueActivateType::Relu): NNOPS_TC_STORE(int(EpilogueActivateType::Relu)); return;
    case int(EpilogueActivateType::Sigmoid):
        NNOPS_TC_STORE(int(EpilogueActivateType::Sigmoid));
        return;
    case int(EpilogueActivateType::Tanh): NNOPS_TC_STORE(int(EpilogueActivateType::Tanh)); return;
    default: NNOPS_TC_STORE(-1); return;
    }
#undef NNOPS_TC_STORE
}

} // namespace nnops::backend::cuda::attn
