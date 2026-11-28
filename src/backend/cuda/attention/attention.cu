/// @file attention.cu
/// @brief CUDA attention — the operator entry points, the parameter fill, and
///        the launch dispatch that picks between the two kernels.
///
/// GroupQueryAttention (ordinary MHA is its H_kv == H case), the cache form and
/// MultiLatentAttention (absorbed and non-absorbed) all map onto one
/// AttentionConfig (backend/common/attention_config.hpp). This file turns that
/// description and the operator's input views into an AttnParams and launches:
///
///   attention_kernel.cuh         the tiled online-softmax kernel — every shape;
///   attention_blocked.cuh        the register-blocked FFMA prefill tile — the
///                                f32 fast path;
///   attention_f16.cuh            the m16n8k16 tensor-core prefill tile — the
///                                16-bit fast path;
///   attention_i8.cuh             the m16n8k32 tile an s8 cache with
///                                cfg.quantize_q takes — the integer route;
///   attention_mla.cuh            the absorbed-MLA tile — one latent row serves
///                                sixteen query heads, and the value is read out
///                                of the key tile itself (FlashMLA's pose);
///   attention_decode.cuh         flash decoding — the one-row (Sq == 1) kernel,
///                                its split-KV form and the merge pass;
///   attention_common.cuh         the shared description, layout strides, K/V
///                                accessors and gate activation.
///
/// The split mirrors cuda/conv (conv2d_common.cuh + per-family .cu) and
/// cuda/matmul (gemm_common.cuh + per-kernel .cuh): the kernels are header-only
/// device code included by this one translation unit, the host side stays here.
///
/// Every kernel shares the CPU core's semantics (see attention_config.hpp): the
/// visible key range is [0, sink) ∪ [max(0, pos_i - window + 1), pos_i + 1)
/// intersected with the batch's run length, the additive bias and keep-mask
/// follow the softcap, the head sink is a virtual zero-valued key in the
/// softmax denominator, and the output gate multiplies the stored value.
///
/// Varlen (packed documents) runs one launch per segment over pitch-preserving
/// sub-views, exactly as the CPU core recurses; no data is copied.

#include "attention_blocked.cuh"
#include "attention_decode.cuh"
#include "attention_kernel.cuh"
#include "attention_f16.cuh"
#include "attention_i8.cuh"
#include "attention_mla.cuh"
#include "attention_mla_host.hpp"
#include "backend/common/attention_config.hpp"
#include "backend/cuda/matmul/gemm_common.cuh" // the split-KV scratch pool
#include "detail/bench_knobs.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

namespace nnops::backend::cuda::attn {

/// Shared-memory bytes a tile of `bc` keys needs: the staged query block, K,
/// V, the score tile (padded row strides on Q, K and S) and the three per-row
/// vectors.
inline int64_t tile_smem_bytes(int bc, int64_t dqk, int64_t dv)
{
    const int64_t kstride = pad_stride(dqk);
    const int64_t sstride = static_cast<int64_t>(bc) + 1;
    return static_cast<int64_t>(sizeof(float)) *
           (static_cast<int64_t>(kBr) * kstride + static_cast<int64_t>(bc) * kstride +
            static_cast<int64_t>(bc) * dv + static_cast<int64_t>(kBr) * sstride + 3 * kBr);
}

/// The largest tile width that fits the device's per-block shared memory,
/// capped at 96 KB. Queried once; a multi-device call pays one probe.
inline int smem_budget()
{
    static int budget = [] {
        int v = 48 * 1024;
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess) {
            int m = 0;
            if (cudaDeviceGetAttribute(&m, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) ==
                cudaSuccess) {
                v = m;
            }
        }
        return std::min(v, 96 * 1024);
    }();
    return budget;
}

/// The multiprocessor count, for the split decision. Queried once per process.
inline int sm_count()
{
    static int sms = [] {
        int dev = 0;
        int n = 1;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
            (void)cudaGetLastError();
            n = 1;
        }
        return std::max(n, 1);
    }();
    return sms;
}

/// How many ways to cut a decode step's key run.
///
/// A decode step is one chain of key rounds per warp, so more blocks is the
/// only way to shorten it — but only up to a single wave. Past that a split
/// repeats the query staging and the partial traffic, and the measured curve
/// turns back up: the GQA row reads 0.152 ms at one split, 0.092 at four,
/// 0.099 at eight and 0.117 at thirty-two, on 38 SMs at the four blocks a
/// multiprocessor this kernel's registers allow (four splits is 128 blocks, a
/// wave). So the target is 85% of one wave, never more.
inline int choose_decode_splits(int64_t rows, int64_t sk, int resident)
{
    const int64_t wave = static_cast<int64_t>(sm_count()) * std::max(resident, 1);
    int splits = 1;
    if (rows > 0 && rows < 2 * wave) {
        splits = static_cast<int>((wave * 17 / 20 + rows / 2) / rows);
    }
    const int forced = ::nnops::detail::cuda_attn_split.load(std::memory_order_relaxed);
    if (forced >= 0) {
        splits = forced;
    } else {
        splits = std::min(splits, kMaxAutoDecodeSplits);
    }
    splits = std::min<int64_t>(splits, std::max<int64_t>(1, sk / kKeysPerRound));
    return std::max(1, std::min(splits, kMaxDecodeSplits));
}

/// Blocks a multiprocessor holds of `kernel` at this shared-memory request.
/// Four is the fallback rather than one: a wrong guess should cost a wave, not
/// five.
template <class K> inline int resident_blocks(K kernel, size_t smem)
{
    int n = 4;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, kThreads, smem) != cudaSuccess) {
        (void)cudaGetLastError();
        n = 4;
    }
    return n < 1 ? 1 : n;
}

/// Point one kernel at a shared-memory size above the 48 KB default. False
/// means the device will not carry the launch.
template <class K> inline bool set_dyn_smem(K kernel, size_t bytes)
{
    if (bytes <= 48 * 1024) {
        return true;
    }
    if (cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(bytes)) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return true;
}

/// The same, plus the largest shared partition. The tensor tile asks for a size
/// two blocks fit in, and the driver otherwise sizes the carveout to the
/// request of one block, so the second never becomes resident. Only that tile
/// takes it: every other attention kernel wants the L1 side of the split.
template <class K> inline bool set_dyn_smem_packed(K kernel, size_t bytes)
{
    if (!set_dyn_smem(kernel, bytes)) {
        return false;
    }
    (void)cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                               cudaFuncAttributePreferredSharedMemoryCarveout,
                               cudaSharedmemCarveoutMaxShared);
    return true;
}

/// Whether the register-blocked tile can carry this call: the shapes its shared
/// budget holds (64 query rows x 128 keys over a head width of 64 to 192 rows
/// of query panel, and a value width the 32 lane column groups cover) and the
/// score features that do not need a per-element gather inside its nest. The
/// knob is the A/B switch: -1 automatic, 0 keeps everything on the tiled
/// kernel, 1 prefers the blocked tile whenever it is eligible.
template <class T, bool KV8>
inline bool blocked_eligible(const AttnParams<T, KV8>& p)
{
    if constexpr (KV8) {
        return false;
    }
    const int mode = ::nnops::detail::cuda_attn_blocked.load(std::memory_order_relaxed);
    if (mode == 0) {
        return false;
    }
    return !p.has_bias && !p.has_keep && !p.softcap_on && p.Dqk <= 192 && p.Dv <= 128 &&
           (p.Dqk % 4) == 0 && (p.Dv % 4) == 0 &&
           blocked::smem_bytes(p.Dqk, p.Dv, blocked::kVsMin) <=
               static_cast<int64_t>(smem_budget());
}

/// Whether the tensor-core tile can carry this call: a 16-bit cache, the score
/// features that need no per-element gather, and a head / value width whose
/// panels sit in the shared budget. The knob is the A/B switch: -1 or 1 take
/// the tile when it is eligible, 0 keeps the FFMA tile (or the tiled kernel).
template <class T, bool KV8>
inline bool tc_eligible(const AttnParams<T, KV8>& p)
{
    if constexpr (KV8 || std::is_same_v<T, float>) {
        return false;
    } else {
        const int mode = ::nnops::detail::cuda_attn_tc.load(std::memory_order_relaxed);
        const int fast = ::nnops::detail::cuda_attn_blocked.load(std::memory_order_relaxed);
        return mode != 0 && fast != 0 && !p.has_bias && !p.has_keep && !p.softcap_on &&
               p.Dqk <= 192 && (p.Dqk % 64) == 0 && (p.Dv % 64) == 0 && p.Dv <= 128;
    }
}

/// Launch the m16n8k16 tile. False means the shape has no instantiation or the
/// device will not carry the shared-memory request; the caller falls back.
template <class T, int DQK, int DV>
inline bool launch_tc_shape(const AttnParams<T, false>& p, const dim3& grid, cudaStream_t stream)
{
    const size_t bytes = static_cast<size_t>(f16::smem_bytes(DQK, DV));
    if (!p.has_gate) {
        if (!set_dyn_smem_packed(&f16::attention_kernel<T, DQK, DV, false>, bytes)) {
            return false;
        }
        f16::attention_kernel<T, DQK, DV, false><<<grid, f16::kThreads, bytes, stream>>>(p);
    } else {
        if (!set_dyn_smem_packed(&f16::attention_kernel<T, DQK, DV, true>, bytes)) {
            return false;
        }
        f16::attention_kernel<T, DQK, DV, true><<<grid, f16::kThreads, bytes, stream>>>(p);
    }
    return cudaGetLastError() == cudaSuccess;
}

template <class T>
inline bool launch_tc(const AttnParams<T, false>& p, cudaStream_t stream)
{
    const size_t budget = static_cast<size_t>(smem_budget());
    const dim3 grid(static_cast<unsigned>(p.H),
                    static_cast<unsigned>(ceil_div(p.Sq, static_cast<int64_t>(f16::kBlockM))),
                    static_cast<unsigned>(p.B));
    auto fits = [&](int64_t dqk, int64_t dv) {
        return static_cast<size_t>(f16::smem_bytes(dqk, dv)) <= budget;
    };

    if (p.Dqk == 128 && p.Dv == 128 && fits(128, 128)) {
        return launch_tc_shape<T, 128, 128>(p, grid, stream);
    } else if (p.Dqk == 64 && p.Dv == 64 && fits(64, 64)) {
        return launch_tc_shape<T, 64, 64>(p, grid, stream);
    } else if (p.Dqk == 192 && p.Dv == 128 && fits(192, 128)) {
        return launch_tc_shape<T, 192, 128>(p, grid, stream);
    }
    return false;
}

/// Whether the integer tile can carry this call: an s8 cache, a query the
/// operator asked to quantize, the score features that need no per-element
/// gather, and one of the (head, value) widths the launcher instantiates. The
/// value product stays 16-bit unless cfg.quantize_pv, which is a query-dtype
/// requirement the tile itself carries: an f32 query has no 16-bit P·V, so it
/// only takes the integer route when both products are integer.
template <class T, bool KV8>
inline bool i8_eligible(const AttnParams<T, KV8>& p)
{
    if constexpr (!KV8) {
        return false;
    } else {
        const int mode = ::nnops::detail::cuda_attn_i8.load(std::memory_order_relaxed);
        if (mode == 0 || !p.quantize_q) {
            return false;
        }
        if constexpr (std::is_same_v<T, float>) {
            if (!p.quantize_pv) {
                return false;
            }
        }
        if (p.has_bias || p.has_keep || p.softcap_on) {
            return false;
        }
        const bool shape = (p.Dqk == 128 && p.Dv == 128) || (p.Dqk == 64 && p.Dv == 64) ||
                           (p.Dqk == 192 && p.Dv == 128);
        if (!shape) {
            return false;
        }
        return p.quantize_pv || (p.Dv % 64) == 0;
    }
}

/// Launch the m16n8k32 tile. False means the device will not carry the
/// shared-memory request; the caller falls back.
template <class T, int DQK, int DV, bool PV8>
inline bool launch_i8_shape(const AttnParams<T, true>& p, const dim3& grid, cudaStream_t stream)
{
    const size_t bytes = static_cast<size_t>(i8::smem_bytes(DQK, DV, PV8));
    if (!set_dyn_smem_packed(&i8::attention_kernel<T, DQK, DV, PV8>, bytes)) {
        return false;
    }
    i8::attention_kernel<T, DQK, DV, PV8><<<grid, i8::kThreads, bytes, stream>>>(p);
    return cudaGetLastError() == cudaSuccess;
}

/// The value product's dtype decides the instantiation: an f32 query has no
/// 16-bit arm (there is no f32 tensor core), so only the integer one exists for
/// it.
template <class T, int DQK, int DV>
inline bool launch_i8_pv(const AttnParams<T, true>& p, const dim3& grid, cudaStream_t stream)
{
    if (p.quantize_pv) {
        return launch_i8_shape<T, DQK, DV, true>(p, grid, stream);
    }
    if constexpr (!std::is_same_v<T, float>) {
        return launch_i8_shape<T, DQK, DV, false>(p, grid, stream);
    }
    return false;
}

template <class T>
inline bool launch_i8(AttnParams<T, true> p, cudaStream_t stream)
{
    const dim3 grid(static_cast<unsigned>(p.H),
                    static_cast<unsigned>(ceil_div(p.Sq, static_cast<int64_t>(i8::kBlockM))),
                    static_cast<unsigned>(p.B));
    const size_t budget = static_cast<size_t>(smem_budget());
    auto fits = [&](int64_t dqk, int64_t dv) {
        return static_cast<size_t>(i8::smem_bytes(dqk, dv, p.quantize_pv)) <= budget;
    };

    // Both value prepasses exist so the tile never transposes or widens a chunk
    // in every block: an fp P·V reads the run widened to 16-bit, the integer one
    // reads it turned over to `[column][key]` (its B panel's pose). The scratch
    // is the run in 16-bit (2 bytes) or in s8 (1 byte), so a cache too large to
    // stage declines the tile rather than paying more than the kernel would.
    const int64_t skt = (p.Sk + 31) & ~static_cast<int64_t>(31);
    detail::GemmWorkspace ws;
    {
        const int64_t bytes = p.quantize_pv
                                  ? p.B * p.Hkv * p.Dv * skt
                                  : p.B * p.Hkv * p.Sk * p.Dv * static_cast<int64_t>(sizeof(T));
        if (bytes > (64LL << 20)) {
            return false;
        }
        ws = detail::gemm_workspace_alloc(bytes, stream);
        if (ws.ptr == nullptr) {
            return false;
        }
        if (p.quantize_pv) {
            p.vt = ws.ptr;
            p.vt_s = skt;
            const dim3 pgrid(static_cast<unsigned>(skt / i8::kBlockN),
                             static_cast<unsigned>(p.Hkv), static_cast<unsigned>(p.B));
            i8::attention_vt_prep_kernel<AttnParams<T, true>>
                <<<pgrid, 128, 0, stream>>>(p, static_cast<int8_t*>(ws.ptr), skt);
        } else {
            p.vprep = ws.ptr;
            const dim3 pgrid(static_cast<unsigned>(p.B * p.Hkv * p.Sk));
            i8::attention_v_prep_kernel<T, AttnParams<T, true>>
                <<<pgrid, 128, 0, stream>>>(p, static_cast<T*>(ws.ptr));
        }
        if (cudaGetLastError() != cudaSuccess) {
            detail::gemm_workspace_free(ws, stream);
            return false;
        }
    }

    bool ok = false;
    if (p.Dqk == 128 && p.Dv == 128 && fits(128, 128)) {
        ok = launch_i8_pv<T, 128, 128>(p, grid, stream);
    } else if (p.Dqk == 64 && p.Dv == 64 && fits(64, 64)) {
        ok = launch_i8_pv<T, 64, 64>(p, grid, stream);
    } else if (p.Dqk == 192 && p.Dv == 128 && fits(192, 128)) {
        ok = launch_i8_pv<T, 192, 128>(p, grid, stream);
    }
    detail::gemm_workspace_free(ws, stream); // stream-ordered when pooled
    return ok;
}

/// The decode: one block per (head, batch, split) over `splits` spans of the
/// run. A one-span call finishes in the kernel; more than one writes partials
/// and a second block-per-(head, batch) kernel merges them.
template <class T, bool KV8>
Status launch_decode(const AttnParams<T, KV8>& p, cudaStream_t stream)
{
    const int64_t rows = p.B * p.H;
    const int hb = std::max(p.head_batch, 1);
    const size_t smem = static_cast<size_t>(decode_smem_bytes(p.Dqk, p.Dv, hb));
    // One output group covers 128 value columns; wider values take the four
    // group form, the narrower the single one. Both are compile-time so the
    // per-lane accumulators stay in registers.
    const bool vec = decode_vec_ok(p);
    const bool wide = (p.Dv + 127) / 128 > 1;
    const int resident =
        vec ? (wide ? resident_blocks(attention_decode_kernel<T, KV8, true, 4>, smem)
                    : resident_blocks(attention_decode_kernel<T, KV8, true, 1>, smem))
            : (wide ? resident_blocks(attention_decode_kernel<T, KV8, false, 4>, smem)
                    : resident_blocks(attention_decode_kernel<T, KV8, false, 1>, smem));
    const int splits = choose_decode_splits(rows / hb, p.Sk, resident);

    DecodeSplits s;
    detail::GemmWorkspace ws;
    if (splits > 1) {
        ws = detail::gemm_workspace_alloc(split_scratch_bytes(rows, splits, p.Dv), stream);
        if (ws.ptr == nullptr) {
            return Status::internal;
        }
        char* base = static_cast<char*>(ws.ptr);
        s.splits = splits;
        s.m = reinterpret_cast<float*>(base);
        s.l = s.m + rows * splits;
        s.o = reinterpret_cast<float*>(base +
                                       rows * splits * 2 * static_cast<int64_t>(sizeof(float)));
    }

    const dim3 dgrid(static_cast<unsigned>(p.H / hb), static_cast<unsigned>(p.B),
                     static_cast<unsigned>(splits));
    if (vec) {
        if (wide) {
            attention_decode_kernel<T, KV8, true, 4><<<dgrid, kThreads, smem, stream>>>(p, s);
        } else {
            attention_decode_kernel<T, KV8, true, 1><<<dgrid, kThreads, smem, stream>>>(p, s);
        }
    } else if (wide) {
        attention_decode_kernel<T, KV8, false, 4><<<dgrid, kThreads, smem, stream>>>(p, s);
    } else {
        attention_decode_kernel<T, KV8, false, 1><<<dgrid, kThreads, smem, stream>>>(p, s);
    }
    if (cudaGetLastError() != cudaSuccess) {
        detail::gemm_workspace_free(ws, stream);
        return Status::internal;
    }

    if (splits > 1) {
        DecodeMergeParams<T> mp;
        mp.B = p.B;
        mp.H = p.H;
        mp.Dv = p.Dv;
        mp.o_b = p.o_b;
        mp.o_h = p.o_h;
        mp.g_b = p.g_b;
        mp.g_h = p.g_h;
        mp.sink = p.sink;
        mp.gate = p.gate;
        mp.o = p.o;
        mp.gate_act = p.gate_act;
        mp.has_gate = p.has_gate;
        const dim3 mgrid(static_cast<unsigned>(p.H), static_cast<unsigned>(p.B));
        attention_decode_merge_kernel<T>
            <<<mgrid, kThreads, kMaxDecodeSplits * sizeof(float), stream>>>(mp, s);
        const bool ok = cudaGetLastError() == cudaSuccess;
        detail::gemm_workspace_free(ws, stream);
        return ok ? Status::ok : Status::internal;
    }
    return Status::ok;
}

// ============================================================
// MultiLatentAttention's absorbed tile
// ============================================================

/// Calls the absorbed tile carried (see attention_mla_host.hpp). One counter a
/// process, like the linear attention route's.
std::atomic<int64_t> g_mla_runs{0};

/// The absorbed tile's shared slab is larger than the 96 KB the other attention
/// kernels are capped at — it stages the whole 576-wide latent row once and
/// reads the value out of that panel — so it asks the device for its own limit
/// rather than the shared cap above. Queried once per process.
inline int mla_smem_budget()
{
    static int budget = [] {
        int v = 48 * 1024;
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) !=
                cudaSuccess) {
            (void)cudaGetLastError();
            v = 48 * 1024;
        }
        return v;
    }();
    return budget;
}

/// Whether the absorbed-MLA tile can carry this call: the latent value form, the
/// score features that need no per-element gather, sixteen query heads under one
/// KV head, and a (key, value) width the launcher instantiates.
template <class T>
inline bool mla_eligible(const AttnParams<T, false>& p)
{
    if constexpr (!std::is_same_v<T, __half>) {
        return false;
    } else {
        const int mode = ::nnops::detail::cuda_attn_mla.load(std::memory_order_relaxed);
        if (mode == 0) {
            return false;
        }
        if (!p.v_is_k || p.has_bias || p.has_keep || p.softcap_on) {
            return false;
        }
        if (p.group < mla::kRows || (p.group % mla::kRows) != 0) {
            return false;
        }
        if (p.Dv % 64 != 0 || p.Dv > p.Dqk || p.Dqk % 32 != 0) {
            return false;
        }
        // The DeepSeek latent only. A narrow absorbed call (192 x 128) belongs
        // to the m16n8k16 tile, which beats this one 2x there (0.082 against
        // 0.153 ms on the 512-row prefill) — the wide value is what this tile
        // is for.
        return p.Dqk == 576 && p.Dv == 512;
    }
}

/// How many ways to cut a decode step's run. A block is one (head group, batch)
/// at one CTA a multiprocessor, so the DeepSeek shape offers eight blocks on
/// this part and the run is the only other axis to spread over.
///
/// The target is one full wave, and an *under*-filled one: every head group of
/// a step reads the same latent run, so blocks resident together share it
/// through L2, while one block over the wave line costs a whole second round of
/// the slowest block. Measured on the 4096-token decode, eight head groups
/// (DeepSeek's 128 heads): four splits, 32 blocks, 0.16 ms; five splits, 40
/// blocks — two over 38 SMs — 0.22 ms; eight splits 0.19. One head group reads
/// 0.084 ms at sixteen splits, 0.097 at thirty-two, so the run-axis split stops
/// at sixteen.
inline int mla_decode_splits(int64_t blocks, int64_t sk)
{
    const int forced = ::nnops::detail::cuda_attn_split.load(std::memory_order_relaxed);
    int splits = 1;
    if (forced >= 0) {
        splits = forced;
    } else if (blocks > 0 && blocks < sm_count()) {
        splits = static_cast<int>(sm_count() / blocks);
    }
    splits = static_cast<int>(std::min<int64_t>(splits, std::max<int64_t>(1, sk / (4 * mla::kBlockN))));
    return std::max(1, std::min(splits, 16));
}

/// Launch the absorbed-MLA tile (and the split merge when the run was cut).
/// False means the shape has no instantiation or the device will not carry the
/// shared-memory request; the caller falls back.
template <class T>
inline bool launch_mla(const AttnParams<T, false>& p, cudaStream_t stream)
{
    // One instantiation, the DeepSeek latent `mla_eligible` gates on; the slab
    // is sized from the same two constants so a widened gate cannot launch a
    // kernel whose panel stride disagrees with its staging.
    constexpr int kDqk = 576, kDv = 512;
    const size_t slab = static_cast<size_t>(mla::smem_bytes(kDqk));
    if (slab > static_cast<size_t>(mla_smem_budget())) {
        return false;
    }
    const int64_t blocks = p.B * (p.H / mla::kRows);
    const int splits = p.Sq == 1 ? mla_decode_splits(blocks, p.Sk) : 1;
    if (p.B * splits > 65535 || p.Sq > 65535) {
        return false;
    }

    mla::MlaSplits s;
    detail::GemmWorkspace ws;
    if (splits > 1) {
        const int64_t rows = p.B * p.H;
        const int64_t bytes = rows * splits * 2 * static_cast<int64_t>(sizeof(float)) +
                              rows * splits * p.Dv * static_cast<int64_t>(sizeof(float));
        ws = detail::gemm_workspace_alloc(static_cast<size_t>(bytes), stream);
        if (ws.ptr == nullptr) {
            return false;
        }
        char* base = static_cast<char*>(ws.ptr);
        s.splits = splits;
        s.m = reinterpret_cast<float*>(base);
        s.l = s.m + rows * splits;
        s.o = reinterpret_cast<float*>(base + rows * splits * 2 * static_cast<int64_t>(sizeof(float)));
    }

    const dim3 grid(static_cast<unsigned>(p.H / mla::kRows), static_cast<unsigned>(p.Sq),
                    static_cast<unsigned>(p.B * splits));
    auto* const kern = &mla::attention_mla_kernel<T, kDqk, kDv>;
    bool ok = false;
    if (set_dyn_smem_packed(kern, slab)) {
        kern<<<grid, mla::kThreads, slab, stream>>>(p, s);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok && splits > 1) {
        const dim3 mgrid(static_cast<unsigned>(p.H / mla::kRows), static_cast<unsigned>(p.B),
                         static_cast<unsigned>(p.Sq));
        mla::attention_mla_merge_kernel<T><<<mgrid, mla::kMergeThreads, 0, stream>>>(p, s);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (splits > 1) {
        detail::gemm_workspace_free(ws, stream);
    }
    if (ok) {
        g_mla_runs.fetch_add(1, std::memory_order_relaxed);
    }
    return ok;
}

/// The dtype/route gate plus the parameter fill. Returns unsupported (rather
/// than a wrong answer) for a value width above the register bound, a shared
/// tile that cannot fit, or an integer route this kernel does not carry.
template <class T, bool KV8>
Status launch_typed(const AttentionConfig& cfg, TensorView& out,
                    std::span<const TensorView> in, cudaStream_t stream)
{
    const TensorView& Q = in[0];
    const TensorView& K = in[1];
    const bool v_is_k = cfg.v_is_k;
    const TensorView& V = v_is_k ? in[1] : in[2];

    // The head geometry is the CPU core's (backend/common/attention_geometry.hpp):
    // same layouts, same widths, same past-segment count.
    const HeadShape hs = resolve_head_shape(cfg, in);
    const HeadLayout ql = hs.q_layout;
    const HeadLayout kl = hs.kv_layout;
    const int64_t B = hs.B;
    const int64_t H = hs.H;
    const int64_t Hkv = hs.Hkv;
    const int64_t Sq = hs.Sq;
    const int64_t Dqk = hs.Dqk;
    const int64_t Dv = hs.Dv;

    const bool has_past = has_past_view(cfg, in);
    const TensorView* P = has_past ? &in[static_cast<size_t>(cfg.past_k_slot)] : nullptr;
    const TensorView* PV = has_past && !v_is_k
                               ? &in[static_cast<size_t>(cfg.past_v_slot)]
                               : P;
    const int64_t Sk_past = hs.Sk_past;
    const int64_t Sk = hs.Sk + hs.Sk_past;

    if (B <= 0 || H <= 0 || Hkv <= 0 || Sq <= 0 || Dqk <= 0 || Dv <= 0 || Sk <= 0) {
        return Status::invalid_argument;
    }
    if (H > 65535 || B > 65535) {
        return Status::unsupported;
    }
    if (Dv > 512) {
        return Status::unsupported; // the per-thread value accumulator is 32 wide
    }

    // The tiled kernel walks 16 lane groups, so its tile is at most 32 keys
    // (two per thread); shrink until it fits the shared-memory budget.
    const int bc = [&] {
        const int budget = smem_budget();
        int w = 32;
        while (w > 8 && tile_smem_bytes(w, Dqk, Dv) > budget) {
            w >>= 1;
        }
        return w;
    }();
    if (tile_smem_bytes(bc, Dqk, Dv) > smem_budget()) {
        return Status::unsupported;
    }
    const size_t smem_bytes = static_cast<size_t>(tile_smem_bytes(bc, Dqk, Dv));

    AttnParams<T, KV8> p;
    p.B = B;
    p.H = H;
    p.Hkv = Hkv;
    p.Sq = Sq;
    p.Dqk = Dqk;
    p.Dv = Dv;
    p.Sk = Sk;
    p.Sk_past = Sk_past;
    p.group = H / Hkv;
    p.scale = cfg.scale > 0.0f ? cfg.scale : 1.0f / std::sqrt(static_cast<float>(Dqk));
    p.bc = bc;

    p.q_b = Q.stride_elems(0);
    p.q_h = head_step(Q, ql, Dqk);
    p.q_s = head_seq_step(Q, ql);

    p.k_b = K.stride_elems(0);
    p.k_h = head_step(K, kl, Dqk);
    p.k_s = head_seq_step(K, kl);
    if (v_is_k) {
        p.v_b = p.k_b;
        p.v_h = p.k_h;
        p.v_s = p.k_s;
    } else {
        p.v_b = V.stride_elems(0);
        p.v_h = head_step(V, kl, Dv);
        p.v_s = head_seq_step(V, kl);
    }
    if (has_past) {
        p.pk_b = P->stride_elems(0);
        p.pk_h = head_step(*P, kl, Dqk);
        p.pk_s = head_seq_step(*P, kl);
        if (v_is_k) {
            p.pv_b = p.pk_b;
            p.pv_h = p.pk_h;
            p.pv_s = p.pk_s;
        } else {
            p.pv_b = PV->stride_elems(0);
            p.pv_h = head_step(*PV, kl, Dv);
            p.pv_s = head_seq_step(*PV, kl);
        }
    }

    p.o_b = out.stride_elems(0);
    p.o_h = head_step(out, ql, Dv);
    p.o_s = head_seq_step(out, ql);

    // ---- masks ----
    if (slot_present(cfg, cfg.bias_slot, in)) {
        const TensorView& m = in[static_cast<size_t>(cfg.bias_slot)];
        p.bias = m.ptr<T>();
        p.has_bias = true;
        if (m.rank() == 2) {
            p.bias_row = m.stride_elems(0);
        } else {
            p.bias_b = m.shape(0) == 1 ? 0 : m.stride_elems(0);
            p.bias_h = m.shape(1) == 1 ? 0 : m.stride_elems(1);
            p.bias_row = m.stride_elems(2);
        }
    }
    if (slot_present(cfg, cfg.kpm_slot, in)) {
        const TensorView& m = in[static_cast<size_t>(cfg.kpm_slot)];
        if (m.rank() >= 2) {
            p.keep = m.ptr<int32_t>();
            p.has_keep = true;
            p.keep_b = m.stride_elems(0);
            p.keep_row = m.rank() == 3 ? m.stride_elems(1) : 0;
        }
    }
    p.filter = cfg.mask_filter_value;

    // ---- per-batch run length: seqlens_k first, else a rank-1 padding length ----
    if (slot_present(cfg, cfg.seqlens_slot, in)) {
        p.run_ptr = in[static_cast<size_t>(cfg.seqlens_slot)].ptr<int32_t>();
        p.run_plus_one = !cfg.seqlens_is_length;
    } else if (slot_present(cfg, cfg.kpm_slot, in) &&
               in[static_cast<size_t>(cfg.kpm_slot)].rank() == 1) {
        p.run_ptr = in[static_cast<size_t>(cfg.kpm_slot)].ptr<int32_t>();
        p.run_plus_one = false;
    }
    if (slot_present(cfg, cfg.pos_ids_slot, in)) {
        p.pos_ids = in[static_cast<size_t>(cfg.pos_ids_slot)].ptr<int64_t>();
    }
    p.position_offset = cfg.position_offset;

    if (slot_present(cfg, cfg.sink_slot, in)) {
        // The sink rides the query's dtype; the kernels read it through
        // `sink_at`.
        p.sink = in[static_cast<size_t>(cfg.sink_slot)].ptr<void>();
    }
    if (slot_present(cfg, cfg.k_scale_slot, in)) {
        const TensorView& s = in[static_cast<size_t>(cfg.k_scale_slot)];
        p.kscale = s.ptr<float>();
        p.scale_per_token = s.numel() > 1;
    }
    if (slot_present(cfg, cfg.v_scale_slot, in)) {
        const TensorView& s = in[static_cast<size_t>(cfg.v_scale_slot)];
        p.vscale = s.ptr<float>();
        p.vscale_per_token = s.numel() > 1;
    }

    // ---- pointers ----
    p.q = Q.ptr<T>();
    p.k = reinterpret_cast<const typename AttnParams<T, KV8>::KVT*>(K.ptr<void>());
    p.v = v_is_k ? p.k
                 : reinterpret_cast<const typename AttnParams<T, KV8>::KVT*>(V.ptr<void>());
    if (has_past) {
        p.pk = reinterpret_cast<const typename AttnParams<T, KV8>::KVT*>(P->ptr<void>());
        p.pv = v_is_k ? p.pk
                      : reinterpret_cast<const typename AttnParams<T, KV8>::KVT*>(PV->ptr<void>());
    }
    p.o = out.ptr<T>();

    p.causal = cfg.causal;
    p.v_is_k = v_is_k;
    p.quantize_q = cfg.quantize_q;
    p.quantize_pv = cfg.quantize_pv;
    p.softcap_on = cfg.softcap > 0.0f;
    p.softcap = cfg.softcap;
    p.window = cfg.window;
    p.sink_tokens = cfg.sink_tokens;

    if (cfg.output_gate != EpilogueActivateType::None) {
        if (!slot_present(cfg, cfg.gate_slot, in)) {
            return Status::invalid_argument;
        }
        const TensorView& gv = in[static_cast<size_t>(cfg.gate_slot)];
        p.gate = gv.ptr<T>();
        p.has_gate = true;
        p.g_b = gv.stride_elems(0);
        p.g_h = head_step(gv, ql, Dv);
        p.g_s = head_seq_step(gv, ql);
        p.gate_act = int(cfg.output_gate);
    }

    // The KEY/VALUE layouts must match the tensor rank they were read from; a
    // past pair follows the K/V order.
    if (has_past && (P->rank() != K.rank() || (!v_is_k && PV->rank() != V.rank()))) {
        return Status::invalid_argument;
    }

    // The absorbed MLA form has its own tile before the general routes: the
    // value is the latent row's prefix, so one staging feeds both products, and
    // the query axis is the head group (see attention_mla.cuh).
    // The constexpr keeps the f32 and bf16 dispatches from instantiating the
    // tile at all: `mla_eligible` can only answer true for half.
    if constexpr (!KV8 && std::is_same_v<T, __half>) {
        if (mla_eligible(p) && launch_mla<T>(p, stream)) {
            return Status::ok;
        }
    }

    // A single query row is the decode shape: all 256 threads work on it, and
    // the run is split across blocks whenever that grid is too small to fill
    // the device.
    if (Sq == 1) {
        p.head_batch = decode_head_batch(p.group, p.Dv);
        return launch_decode<T, KV8>(p, stream);
    }

    if constexpr (KV8) {
        // An s8 cache the operator asked to quantize against runs the integer
        // tile; every other s8 call keeps widening on read through the tiled
        // kernel below (the FFMA and 16-bit tiles take an fp cache only).
        if (i8_eligible(p) && launch_i8<T>(p, stream)) {
            return Status::ok;
        }
    } else {
        // The 16-bit cache goes to the tensor cores first; the FFMA tile is its
        // fallback and the f32 route. (The constexpr keeps the f32 T from
        // instantiating the m16n8k16 kernels at all.)
        if constexpr (!std::is_same_v<T, float>) {
            if (tc_eligible(p) && launch_tc<T>(p, stream)) {
                return Status::ok;
            }
        }
        if (blocked_eligible(p)) {
            // The staged V slice is what keeps the PV nest off global memory.
            // It is as deep as the shared budget holds: 32 keys beside a
            // 128-wide query panel, 16 beside a 192-wide one, 8 if even that
            // will not fit. Every slice costs two barriers a tile, so the
            // depths are worth stepping down in tens rather than eights.
            const size_t budget = static_cast<size_t>(smem_budget());
            const dim3 bgrid(
                static_cast<unsigned>(H),
                static_cast<unsigned>(ceil_div(Sq, static_cast<int64_t>(blocked::kBlockM))),
                static_cast<unsigned>(B));
            auto fits = [&](int kvs) {
                return static_cast<size_t>(blocked::smem_bytes(Dqk, Dv, kvs)) <= budget;
            };
            const int kvs = fits(blocked::kVsMax) ? blocked::kVsMax
                                                  : (fits(blocked::kVsMid) ? blocked::kVsMid
                                                                           : blocked::kVsMin);
            const size_t bsmem = static_cast<size_t>(blocked::smem_bytes(Dqk, Dv, kvs));
            if (kvs == blocked::kVsMax) {
                if (set_dyn_smem(&blocked::attention_blocked_kernel<T, blocked::kVsMax>, bsmem)) {
                    blocked::attention_blocked_kernel<T, blocked::kVsMax>
                        <<<bgrid, blocked::kThreads, bsmem, stream>>>(p);
                    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
                }
            } else if (kvs == blocked::kVsMid) {
                if (set_dyn_smem(&blocked::attention_blocked_kernel<T, blocked::kVsMid>, bsmem)) {
                    blocked::attention_blocked_kernel<T, blocked::kVsMid>
                        <<<bgrid, blocked::kThreads, bsmem, stream>>>(p);
                    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
                }
            } else if (set_dyn_smem(&blocked::attention_blocked_kernel<T, blocked::kVsMin>, bsmem)) {
                blocked::attention_blocked_kernel<T, blocked::kVsMin>
                    <<<bgrid, blocked::kThreads, bsmem, stream>>>(p);
                return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
            }
        }
    }

    if (!set_dyn_smem(&attention_kernel<T, KV8>, smem_bytes)) {
        return Status::unsupported;
    }

    const dim3 grid(static_cast<unsigned>(ceil_div(Sq, static_cast<int64_t>(kBr))),
                    static_cast<unsigned>(H), static_cast<unsigned>(B));
    const dim3 block(kThreads);
    attention_kernel<T, KV8><<<grid, block, smem_bytes, stream>>>(p);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

/// Dispatch on the query dtype (and whether the cache is an s8 one).
Status attention_launch(const AttentionConfig& cfg, TensorView& out,
                        std::span<const TensorView> in, cudaStream_t stream)
{
    if (in.empty() || out.is_empty()) {
        return Status::invalid_argument;
    }
    const TensorView& Q = in[0];
    if (in.size() < 2 || in[1].is_empty()) {
        return Status::invalid_argument;
    }
    const bool v_is_k = cfg.v_is_k;
    if (!v_is_k && (in.size() < 3 || in[2].is_empty())) {
        return Status::invalid_argument;
    }

    const DataType dt = Q.data_type();
    const DataType kdt = in[1].data_type();
    const bool kv8 = kdt == DataType::s8;
    if (kv8) {
        if (dt != DataType::f32 && dt != DataType::f16 && dt != DataType::bf16) {
            return Status::unsupported;
        }
        if (!v_is_k && in[2].data_type() != DataType::s8) {
            return Status::unsupported;
        }
    } else if (kdt != dt || (!v_is_k && in[2].data_type() != dt)) {
        return Status::unsupported;
    }
    if (out.data_type() != dt) {
        return Status::invalid_argument;
    }

    // The gate rides the output, so its dtype and geometry are the output's.
    if (cfg.output_gate != EpilogueActivateType::None) {
        if (!slot_present(cfg, cfg.gate_slot, in)) {
            return Status::invalid_argument;
        }
        const TensorView& gv = in[static_cast<size_t>(cfg.gate_slot)];
        if (gv.data_type() != dt || gv.rank() != out.rank()) {
            return Status::invalid_argument;
        }
    }

    switch (dt) {
    case DataType::f32:
        return kv8 ? launch_typed<float, true>(cfg, out, in, stream)
                   : launch_typed<float, false>(cfg, out, in, stream);
    case DataType::f16:
        return kv8 ? launch_typed<__half, true>(cfg, out, in, stream)
                   : launch_typed<__half, false>(cfg, out, in, stream);
    case DataType::bf16:
        return kv8 ? launch_typed<__nv_bfloat16, true>(cfg, out, in, stream)
                   : launch_typed<__nv_bfloat16, false>(cfg, out, in, stream);
    default:
        return Status::unsupported;
    }
}

// ============================================================
// Varlen: one launch per packed segment over pitch-preserving views
// ============================================================

inline TensorView packed_row_view(const TensorView& parent, int64_t row_off, int64_t rows,
                                  int64_t width)
{
    const int64_t dims[3] = {1, rows, width};
    char* base = const_cast<char*>(parent.ptr<char>()) + row_off * parent.pitch();
    return TensorView(std::span<const int64_t>(dims, 3), parent.data_type(), base, parent.pitch(),
                      parent.layout());
}

Status attention_varlen(const AttentionConfig& cfg, TensorView& out,
                        std::span<const TensorView> in, cudaStream_t stream)
{
    if (cfg.position_offset >= 0) {
        return Status::invalid_argument;
    }
    if (!slot_present(cfg, cfg.cu_q_slot, in) || !slot_present(cfg, cfg.cu_k_slot, in) ||
        slot_present(cfg, cfg.past_k_slot, in) || slot_present(cfg, cfg.seqlens_slot, in) ||
        slot_present(cfg, cfg.pos_ids_slot, in) ||
        (slot_present(cfg, cfg.bias_slot, in) && in[static_cast<size_t>(cfg.bias_slot)].rank() >= 2) ||
        (slot_present(cfg, cfg.kpm_slot, in) &&
         in[static_cast<size_t>(cfg.kpm_slot)].rank() >= 2)) {
        return Status::invalid_argument;
    }
    const auto& Q = in[0];
    const auto& K = in[1];
    const auto& V = cfg.v_is_k ? in[1] : in[2];
    if (Q.rank() != 3 || K.rank() != 3 || V.rank() != 3 || Q.shape(0) != 1) {
        return Status::invalid_argument;
    }

    const auto& cq = in[static_cast<size_t>(cfg.cu_q_slot)];
    const auto& ck = in[static_cast<size_t>(cfg.cu_k_slot)];
    if (cq.data_type() != DataType::s32 || ck.data_type() != DataType::s32 || cq.rank() != 1 ||
        ck.rank() != 1 || cq.shape(0) != ck.shape(0) || cq.shape(0) < 2) {
        return Status::invalid_argument;
    }
    const int64_t nseg = cq.shape(0) - 1;
    const int64_t total_q = Q.shape(1);
    const int64_t total_k = K.shape(1);

    // The offsets drive the host-side segment loop, so they have to be readable
    // here: the views point into device memory, so copy them down in one go.
    std::vector<int32_t> hq(static_cast<size_t>(nseg + 1)), hk(static_cast<size_t>(nseg + 1));
    if (cudaMemcpy(hq.data(), cq.ptr<int32_t>(), hq.size() * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(hk.data(), ck.ptr<int32_t>(), hk.size() * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        (void)cudaGetLastError();
        return Status::internal;
    }
    const int32_t* qo = hq.data();
    const int32_t* ko = hk.data();

    std::vector<TensorView> seg(in.begin(), in.end());
    for (int64_t s = 0; s < nseg; ++s) {
        const int64_t q0 = qo[s], q1 = qo[s + 1];
        const int64_t k0 = ko[s], k1 = ko[s + 1];
        if (q0 < 0 || q1 < q0 || q1 > total_q || k0 < 0 || k1 < k0 || k1 > total_k) {
            return Status::invalid_argument;
        }
        const int64_t len_q = q1 - q0;
        const int64_t len_k = k1 - k0;
        if (cfg.causal && len_k < len_q) {
            return Status::invalid_argument;
        }
        if (len_q == 0) {
            continue;
        }

        seg[0] = packed_row_view(Q, q0, len_q, Q.shape(2));
        seg[1] = packed_row_view(K, k0, len_k, K.shape(2));
        if (!cfg.v_is_k) {
            seg[2] = packed_row_view(V, k0, len_k, V.shape(2));
        }
        if (slot_present(cfg, cfg.gate_slot, seg)) {
            seg[static_cast<size_t>(cfg.gate_slot)] =
                packed_row_view(in[static_cast<size_t>(cfg.gate_slot)], q0, len_q,
                                in[static_cast<size_t>(cfg.gate_slot)].shape(2));
        }
        TensorView os = packed_row_view(out, q0, len_q, out.shape(2));
        const Status st = attention_launch(cfg, os, seg, stream);
        if (st != Status::ok) {
            return st;
        }
    }
    return Status::ok;
}

} // namespace nnops::backend::cuda::attn

// ============================================================
// Operator entry points
// ============================================================

namespace nnops::backend::cuda {

int64_t attention_mla_runs() { return attn::g_mla_runs.load(std::memory_order_relaxed); }

Status attention_cuda(const AttentionConfig& cfg, std::span<TensorView> outputs,
                      std::span<const TensorView> inputs, const ComputeContext& ctx, void*)
{
    if (outputs.empty() || outputs[0].is_empty()) {
        return Status::invalid_argument;
    }
    const cudaStream_t stream = static_cast<cudaStream_t>(ctx.cuda_stream);
    TensorView& out = outputs[0];
    return slot_present(cfg, cfg.cu_q_slot, inputs)
               ? attn::attention_varlen(cfg, out, inputs, stream)
               : attn::attention_launch(cfg, out, inputs, stream);
}

Status group_query_attention_cuda(const GroupQueryAttentionAttributes& attrs,
                                  std::span<TensorView> outputs, std::span<const TensorView> inputs,
                                  const ComputeContext& ctx, void* workspace)
{
    return attention_cuda(make_attention_config(attrs), outputs, inputs, ctx, workspace);
}

Status multi_latent_attention_cuda(const MultiLatentAttentionAttributes& attrs, TensorView& output,
                                   std::span<const TensorView> inputs, const ComputeContext& ctx,
                                   void* workspace)
{
    std::span<TensorView> outputs(&output, 1);
    return attention_cuda(make_attention_config(attrs), outputs, inputs, ctx, workspace);
}

} // namespace nnops::backend::cuda
