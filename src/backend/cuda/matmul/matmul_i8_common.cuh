/// @file matmul_i8_common.cuh
/// @brief The integer GEMM's contract: the parameter struct, the zero-point
///        compensation, the requant, the per-element epilogue, the two
///        reduction prepasses, and the byte transpose fill.
///
/// Everything here is shared by the two int8 engines — the hand-written PTX
/// tile and the narrow-m `__dp4a` kernel — so they cannot drift apart. The
/// fill takes a plain pointer and a row stride, which is what lets either
/// kernel include it without any tile machinery.
///
/// MatMulInteger semantics over the SM80 16x8x32 s8 tensor-core atom. A is s8 or
/// u8 and B is s8; a u8 A takes `u8.s8` so the raw accumulate is the true integer
/// dot product in A's own storage domain, and no pack shift is left behind:
///
///     v = raw − zp_a·R_b[n] − (R_a[m] − K·zp_a)·zp_b[n]
///
/// R_a[m] = Σ_k qa[m,k] and R_b[n] = Σ_k qb[k,n] are swept per batch element by
/// the prepass kernels below.

#pragma once

#include "gemm_common.cuh"
#include "nnops/core/data_type.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace nnops::backend::cuda {

/// Where the compensated int32 accumulate goes. BF16 is the closed set's one
/// extra: no operator resolves it as an output dtype (MatMul's integer pair
/// dequantizes to f32/f16), but a recurrent layer's compute dtype may be bf16,
/// and its gate buffers are in that dtype.
enum class I8OutKind {
    S32,
    S8,
    U8,
    F32,
    F16,
    BF16
};

/// Map an output dtype onto the kind the epilogue switches on.
inline bool i8_out_kind(DataType dt, I8OutKind& kind) noexcept
{
    switch (dt) {
    case DataType::s32:
        kind = I8OutKind::S32;
        return true;
    case DataType::s8:
        kind = I8OutKind::S8;
        return true;
    case DataType::u8:
        kind = I8OutKind::U8;
        return true;
    case DataType::f32:
        kind = I8OutKind::F32;
        return true;
    case DataType::f16:
        kind = I8OutKind::F16;
        return true;
    default:
        return false;
    }
}

/// Everything the int8 GEMM needs besides the element type and the two transpose
/// flags. `g` carries the geometry, the batch strides, the output pointer/pitch
/// and the epilogue contract (clip + activation, read by the dequantizing path).
struct GemmI8Params {
    GemmParams g;

    /// A is [M, K] k-contiguous (no transpose_a); B is [N, K] k-contiguous
    /// (transpose_b). The prepasses and the gathers both need this orientation,
    /// which is a property of the call rather than of the transpose template.
    bool a_row_major = true;
    bool b_row_major = true;
    /// Base pointer and row pitch are 16 B aligned, so the cp.async path is
    /// legal. Only meaningful for a k-contiguous operand.
    bool a_vec = false;
    bool b_vec = false;

    // Per-batch-element reductions: [batch][M] and [batch][N].
    int32_t* r_a = nullptr;
    int32_t* r_b = nullptr;

    // Quantization parameters. The per-token (M) and per-channel (N) arrays are
    // device copies staged by the launcher; the scalars are the per-tensor
    // fallback. A's parameters live in A's storage domain (a u8 zero point is
    // 0..255), the domain the u8 MMA accumulates in.
    const int32_t* zp_a = nullptr;
    int32_t zp_a_scalar = 0;
    const int32_t* zp_b = nullptr;
    int32_t zp_b_scalar = 0;
    const float* sc_a = nullptr;
    float sc_a_scalar = 1.0f;
    const float* sc_b = nullptr;
    float sc_b_scalar = 1.0f;
    /// `sc_a`/`zp_a` already point at device memory (a per-row table the
    /// quantization kernel's caller staged or wrote), so the launcher must not
    /// stage them from the host.
    bool sc_a_device = false;
    bool zp_a_device = false;

    /// The requant destination's per-tensor parameters (s8/u8 output only).
    float out_scale = 1.0f;
    int32_t out_zp = 0;

    I8OutKind out = I8OutKind::S32;
    const void* bias = nullptr; ///< [N]; int32 for an integer destination
    bool relu = false;          ///< integer destinations: Relu before the store

    /// The A domain the accumulate lives in: 0 when the MMA sees the stored
    /// values (the tensor-core path, whose u8 atom is unsigned), -128 when the
    /// `__dp4a` narrow path had to sign-shift a u8 A. It enters the compensation
    /// as zp_a + a_shift and r_a + a_shift*K, so the result is the same.
    int32_t a_shift = 0;
};

namespace detail {

template <class Elem> __device__ inline float i8_to_float(Elem x) { return float(x); }
template <> __device__ inline float i8_to_float<float>(float x) { return x; }

/// round(x) with ties away from zero, the scalar/vector epilogue's rule.
__device__ inline float i8_round_away(float x) { return truncf(x + copysignf(0.5f, x)); }

/// The compensated value, exactly as `matmul_int8_compensate` expresses it:
/// raw − zp_a·R_b[n] − (R_a[m] − K·zp_a)·zp_b[n]. `a_shift` moves the whole
/// formula into the domain the accumulator was built in (0 for the u8 MMA, -128
/// for a sign-shifted `__dp4a` A). The two reduction planes arrive already
/// offset to this batch element, so the loop body is pure 32-bit arithmetic.
__device__ inline int32_t i8_compensate(GemmI8Params q, int32_t raw, int m, int n,
                                        const int32_t* r_a_b, const int32_t* r_b_b)
{
    const int32_t zpa = (q.zp_a != nullptr ? q.zp_a[m] : q.zp_a_scalar) + q.a_shift;
    const int32_t zpb = q.zp_b != nullptr ? q.zp_b[n] : q.zp_b_scalar;
    const int32_t ra_v = r_a_b[m] + q.a_shift * q.g.k;
    int32_t v = raw - zpa * r_b_b[n];
    v -= (ra_v - q.g.k * zpa) * zpb;
    return v;
}

/// The s8/u8 requant, spelled so the f32 product, the clamp, the rounding and
/// the convert match `matmul_int8_requant`.
template <class TOut>
__device__ inline TOut i8_requant(int32_t acc, float row_k, float s_b, float zp_out)
{
    constexpr float kMin = std::is_same_v<TOut, uint8_t> ? 0.0f : -128.0f;
    constexpr float kMax = std::is_same_v<TOut, uint8_t> ? 255.0f : 127.0f;
    const float req = row_k * s_b;
    const float zpf = zp_out;
    float x = float(acc) * req;
    x = fminf(fmaxf(x, kMin - zpf), kMax - zpf);
    return static_cast<TOut>(int32_t(i8_round_away(x)) + int32_t(zpf));
}

/// The integer destination's value: the compensated accumulate, the int32 bias,
/// and the Relu stage an integer destination applies (the only one it does).
__device__ inline int32_t i8_int_value(const GemmI8Params& q, int32_t raw, int m, int n,
                                       const int32_t* r_a_b, const int32_t* r_b_b)
{
    int32_t v = i8_compensate(q, raw, m, n, r_a_b, r_b_b);
    const int32_t* bias32 = static_cast<const int32_t*>(q.bias);
    if (bias32 != nullptr) {
        v += bias32[n];
    }
    if (q.relu) {
        v = v < 0 ? 0 : v;
    }
    return v;
}

/// The s8/u8 requant of that value -- so the argument is the biased, Relu'd int,
/// not the raw accumulate. Spelled so the f32 product, the clamp, the rounding
/// and the convert match `matmul_int8_requant` on the CPU.
__device__ inline int32_t i8_requant_value(const GemmI8Params& q, int32_t v, int m, int n)
{
    const float s_am = q.sc_a != nullptr ? q.sc_a[m] : q.sc_a_scalar;
    const float s_bn = q.sc_b != nullptr ? q.sc_b[n] : q.sc_b_scalar;
    const float row_k = s_am / q.out_scale;
    return q.out == I8OutKind::S8 ? int32_t(i8_requant<int8_t>(v, row_k, s_bn, float(q.out_zp))) :
                                    int32_t(i8_requant<uint8_t>(v, row_k, s_bn, float(q.out_zp)));
}

/// The dequantizing destination's value: the compensated accumulate scaled by
/// s_a*s_b, the bias in the output's own format, the clamp and the activation.
/// The bias lands *after* the dequantization here, unlike an integer
/// destination's, which is why this cannot share the value above.
__device__ inline float i8_fp_value(const GemmI8Params& q, int32_t raw, int m, int n,
                                    const int32_t* r_a_b, const int32_t* r_b_b)
{
    const GemmParams& p = q.g;
    const int32_t v = i8_compensate(q, raw, m, n, r_a_b, r_b_b);
    const float s_am = q.sc_a != nullptr ? q.sc_a[m] : q.sc_a_scalar;
    const float s_bn = q.sc_b != nullptr ? q.sc_b[n] : q.sc_b_scalar;
    float f = i8_to_float(v) * (s_am * s_bn);
    float b = 0.0f;
    if (q.out == I8OutKind::F32) {
        const float* biasf = static_cast<const float*>(q.bias);
        if (biasf != nullptr) {
            b = biasf[n];
        }
    } else if (q.out == I8OutKind::BF16) {
        const __nv_bfloat16* biasb = static_cast<const __nv_bfloat16*>(q.bias);
        if (biasb != nullptr) {
            b = __bfloat162float(biasb[n]);
        }
    } else {
        const __half* biash = static_cast<const __half*>(q.bias);
        if (biash != nullptr) {
            b = __half2float(biash[n]);
        }
    }
    // A zero bias is skipped: x + 0 turns -0 into +0, and fp outputs are
    // compared bitwise across the two backends.
    if (b != 0.0f) {
        f += b;
    }
    f = fminf(fmaxf(f, p.min_clip), p.max_clip);
    return gemm_apply_epilogue(p.epilogue, p.ep_alpha, p.ep_beta, f);
}

/// One output element from its compensated accumulate: the value path its
/// destination calls for, then the store. Shared by the tile engines, the
/// narrow-m kernel and the split-K reduce so they cannot drift apart.
__device__ inline void i8_epilogue_element(GemmI8Params q, int64_t c_off, const int32_t* r_a_b,
                                           const int32_t* r_b_b, int32_t raw, int m, int n)
{
    const GemmParams& p = q.g;
    const int64_t idx = int64_t(m) * p.ldc + n;
    switch (q.out) {
    case I8OutKind::S32:
        static_cast<int32_t*>(p.c)[c_off + idx] = i8_int_value(q, raw, m, n, r_a_b, r_b_b);
        return;
    case I8OutKind::S8:
        static_cast<int8_t*>(p.c)[c_off + idx] =
            int8_t(uint8_t(i8_requant_value(q, i8_int_value(q, raw, m, n, r_a_b, r_b_b), m, n)));
        return;
    case I8OutKind::U8:
        static_cast<uint8_t*>(p.c)[c_off + idx] =
            uint8_t(i8_requant_value(q, i8_int_value(q, raw, m, n, r_a_b, r_b_b), m, n));
        return;
    case I8OutKind::F32:
        static_cast<float*>(p.c)[c_off + idx] = i8_fp_value(q, raw, m, n, r_a_b, r_b_b);
        return;
    case I8OutKind::F16:
        static_cast<__half*>(p.c)[c_off + idx] =
            __float2half(i8_fp_value(q, raw, m, n, r_a_b, r_b_b));
        return;
    case I8OutKind::BF16:
        static_cast<__nv_bfloat16*>(p.c)[c_off + idx] =
            __float2bfloat16(i8_fp_value(q, raw, m, n, r_a_b, r_b_b));
        return;
    }
}

/// Block-wide sum of `s`, written by lane 0.
__device__ inline void i8_block_sum_store(int32_t s, int32_t* dst)
{
    __shared__ int32_t red[32];
    const int lane = int(threadIdx.x) & 31;
    const int wid = int(threadIdx.x) >> 5;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
    }
    if (lane == 0) {
        red[wid] = s;
    }
    __syncthreads();
    if (wid == 0) {
        s = (lane < int(blockDim.x) / 32) ? red[lane] : 0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_down_sync(0xffffffffu, s, off);
        }
        if (lane == 0) {
            *dst = s;
        }
    }
}

/// Σ_k of one A row, per batch element. One block per (m, batch).
template <class AT> static __global__ void i8_row_sums_kernel(GemmI8Params q, int32_t* out)
{
    const GemmParams& p = q.g;
    const int batch = int(blockIdx.y);
    const int m = int(blockIdx.x);
    const AT* a = static_cast<const AT*>(p.a) + gemm_batch_offset(p, p.a_batch, batch);

    int32_t s = 0;
    if (q.a_row_major) {
        for (int k = int(threadIdx.x); k < p.k; k += int(blockDim.x)) {
            s += int32_t(a[int64_t(m) * p.lda + k]);
        }
    } else {
        for (int k = int(threadIdx.x); k < p.k; k += int(blockDim.x)) {
            s += int32_t(a[int64_t(m) + int64_t(k) * p.lda]);
        }
    }

    i8_block_sum_store(s, out + int64_t(batch) * p.m + m);
}

/// Σ_k of one B column, per batch element. A k-contiguous B `[N, K]` takes one
/// block per column (coalesced). A `[K, N]` B is walked the other way: four
/// columns per thread, so a warp reads 128 consecutive bytes of each row, and
/// the k axis is cut across blockIdx.z with atomic adds into a zeroed plane.
/// Column-at-a-time here would put every byte behind a stride-`ldb` sector.
static __global__ void i8_col_sums_kernel(GemmI8Params q, int32_t* out)
{
    const GemmParams& p = q.g;
    const int batch = int(blockIdx.y);
    const int8_t* b = static_cast<const int8_t*>(p.b) + gemm_batch_offset(p, p.b_batch, batch);

    if (q.b_row_major) {
        const int n = int(blockIdx.x);
        int32_t s = 0;
        for (int k = int(threadIdx.x); k < p.k; k += int(blockDim.x)) {
            s += int32_t(b[int64_t(n) * p.ldb + k]);
        }
        i8_block_sum_store(s, out + int64_t(batch) * p.n + n);
        return;
    }

    constexpr int kV = 4;
    const int n0 = int(blockIdx.x) * int(blockDim.x) * kV + int(threadIdx.x) * kV;
    int32_t* dst = out + int64_t(batch) * p.n;
    const bool row_aligned = (p.ldb % 4 == 0) && (reinterpret_cast<uintptr_t>(b) % 4 == 0);
    // Every blockIdx.z slice owns a k range -- the tail path included, or each
    // slice would add the whole column.
    const int zs = int(gridDim.z);
    const int z = int(blockIdx.z);
    const int k_begin = int(int64_t(p.k) * z / zs);
    const int k_end = int(int64_t(p.k) * (z + 1) / zs);
    if (n0 + kV <= p.n && row_aligned) {
        int32_t s[kV] = {0, 0, 0, 0};
        for (int k = k_begin; k < k_end; ++k) {
            const uint32_t v = *reinterpret_cast<const uint32_t*>(b + int64_t(k) * p.ldb + n0);
            const int8_t* e = reinterpret_cast<const int8_t*>(&v);
#pragma unroll
            for (int t = 0; t < kV; ++t) {
                s[t] += int32_t(e[t]);
            }
        }
#pragma unroll
        for (int t = 0; t < kV; ++t) {
            atomicAdd(dst + n0 + t, s[t]);
        }
        return;
    }
#pragma unroll
    for (int t = 0; t < kV; ++t) {
        const int n = n0 + t;
        if (n >= p.n) {
            break;
        }
        int32_t s = 0;
        for (int k = k_begin; k < k_end; ++k) {
            s += int32_t(b[int64_t(k) * p.ldb + n]);
        }
        atomicAdd(dst + n, s);
    }
}

/// Fill a k-major s8 panel row block from a source whose contiguous run is along
/// the other axis: read 4-byte words (four consecutive columns at one row),
/// transpose the 4x4 byte blocks in registers, and store 4-byte words (four
/// consecutive k at one panel row). The panel is k-major at exactly that
/// granularity, so the bytes land where the MMA reads them -- and a word store
/// is one instruction where the byte scatter is four.
///
/// `dst` is the stage's panel base and `ld` its row stride in bytes, so the
/// caller keeps its own layout: a strided view and a plain pointer both work.
/// A 4-byte-aligned source is the caller's
/// contract (the word reads below assume it).
template <int bN, int bK>
__device__ inline void i8_transpose_fill(int8_t* dst, int ld, const int8_t* b, int n0, int k0,
                                         int N, int K, int ldb, bool word, int tid, int nthr)
{
    constexpr int nb4 = bN / 4;
    constexpr int nblocks = nb4 * (bK / 4);
    for (int idx = tid; idx < nblocks; idx += nthr) {
        const int k4 = idx / nb4;
        const int n4 = idx - k4 * nb4;
        const int n = n0 + n4 * 4;
        const int k = k0 + k4 * 4;
        int8_t* const row = dst + (n4 * 4) * ld + k4 * 4;
        // The 4 B reads below need a word-addressable base and pitch; a source
        // that is not takes the byte scatter, which is why this is a runtime
        // flag rather than a template.
        if (word && n + 4 <= N && k + 4 <= K) {
            const int8_t* p = b + int64_t(k) * ldb + n;
            const uint32_t w0 = *reinterpret_cast<const uint32_t*>(p);
            const uint32_t w1 = *reinterpret_cast<const uint32_t*>(p + ldb);
            const uint32_t w2 = *reinterpret_cast<const uint32_t*>(p + 2 * ldb);
            const uint32_t w3 = *reinterpret_cast<const uint32_t*>(p + 3 * ldb);
            const uint32_t u0 = __byte_perm(w0, w1, 0x40);
            const uint32_t u1 = __byte_perm(w0, w1, 0x51);
            const uint32_t u2 = __byte_perm(w0, w1, 0x62);
            const uint32_t u3 = __byte_perm(w0, w1, 0x73);
            const uint32_t v0 = __byte_perm(w2, w3, 0x40);
            const uint32_t v1 = __byte_perm(w2, w3, 0x51);
            const uint32_t v2 = __byte_perm(w2, w3, 0x62);
            const uint32_t v3 = __byte_perm(w2, w3, 0x73);
            *reinterpret_cast<uint32_t*>(row) = __byte_perm(u0, v0, 0x5410);
            *reinterpret_cast<uint32_t*>(row + ld) = __byte_perm(u1, v1, 0x5410);
            *reinterpret_cast<uint32_t*>(row + 2 * ld) = __byte_perm(u2, v2, 0x5410);
            *reinterpret_cast<uint32_t*>(row + 3 * ld) = __byte_perm(u3, v3, 0x5410);
            continue;
        }
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                const int nn = n + i;
                const int kk = k + j;
                row[i * ld + j] =
                    (nn < N && kk < K) ? int8_t(b[int64_t(kk) * ldb + nn]) : int8_t(0);
            }
        }
    }
}

} // namespace detail

/// A 16 B cp.async needs both the base pointer and the row pitch aligned; the
/// k and batch offsets are multiples of the row pitch, so those follow.
inline bool i8_operand_vectorizable(const void* ptr, int64_t ld)
{
    return (reinterpret_cast<uintptr_t>(ptr) % 16 == 0) && (ld % 16 == 0);
}

} // namespace nnops::backend::cuda
