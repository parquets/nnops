#pragma once
/// @file conv3d_common.cuh
/// @brief The 3-D convolution family's kernel: one implicit GEMM over the
///        output-voxel axis, with the transposed pose as a gather predicate.
///
/// Conv3D and TransposeConv3D are the same product. Both write an output voxel
/// from a reduction over (ic, kd, kh, kw); the difference is which input voxel
/// a (voxel, tap) pair names — Conv3D strides forward from the output position,
/// TransposeConv3D divides (o + pad − k·dil) by the stride and drops the taps
/// that do not divide. Everything else — the operand walks, the panel fills,
/// the epilogue — is shared, so it is one kernel with a template flag.
///
/// M is the output-voxel axis, N the output channels of one group, K the
/// reduction IC/G·KD·KH·KW. One block owns 64 voxels by 64 channels, sixteen a
/// thread in a 4x4 register tile, and steps the reduction 32 k at a time. The
/// activation tile is the gathered operand: its panels are k-major so the inner
/// loop reads one float4 from each of A and B per four-by-four of the tile, and
/// the fill is m-fastest inside a k so a warp's reads are one tap's channels,
/// which are the contiguous ones.
///
/// The k index decodes to a tap and an input channel, which for Conv3D's DHWIO
/// filter is also the filter's own innermost order; both filter orders are
/// walked through their strides, so a padded or grouped filter reads correctly
/// and one expression serves DHWIO and TransposeConv3D's [IC][OC/G][KD][KH][KW].
///
/// The decode is done once per k into shared memory rather than once per
/// (voxel, k): the 64 rows of a tile share it, and a division per element is
/// most of what the fill would otherwise cost.

#include "backend/cuda/cuda_common.cuh"
#include "backend/cuda/matmul/matmul_epilogue.cuh" // gemm_apply_activation
#include "nnops/core/status.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace nnops::backend::cuda::conv3d {

constexpr int kC3Threads = 256;
constexpr int kC3BM = 64; ///< output voxels a block owns
constexpr int kC3BN = 64; ///< output channels a block owns
constexpr int kC3BK = 32; ///< reduction per step
constexpr int kC3TM = 4;  ///< voxels a thread computes
constexpr int kC3TN = 4;  ///< channels a thread computes
/// Panel row stride: a multiple of four elements so the inner loop's float4 read
/// is aligned, and 4 past the row so the sixteen lanes of a row spread over
/// eight bank groups twice rather than thirty-two once.
constexpr int kC3SA = kC3BM + 4;
constexpr int kC3SB = kC3BN + 4;

/// One 3-D convolution's geometry, strides and epilogue. The input points at its
/// batch slice and the output at its batch·group slice; the filter at the
/// group's base.
struct C3Params {
    int id = 0, ih = 0, iw = 0, ic = 0;
    int od = 0, oh = 0, ow = 0, oc = 0;
    int kd = 1, kh = 1, kw = 1;
    int sd = 1, sh = 1, sw = 1;
    int dd = 1, dh = 1, dw = 1;
    int pd = 0, ph = 0, pw = 0;
    int icg = 1, ocg = 1, groups = 1;

    int64_t m = 0; ///< OD·OH·OW: the M axis of one batch·group slice
    int64_t in_batch = 0, in_d = 0, in_h = 0, in_w = 0, in_c = 0;
    int64_t out_batch = 0, out_d = 0, out_h = 0, out_w = 0, out_c = 0;
    int64_t w_kd = 0, w_kh = 0, w_kw = 0, w_ic = 0, w_oc = 0;
    int64_t w_group = 0; ///< element stride between groups

    const void* in = nullptr;
    const void* w = nullptr;
    void* out = nullptr;
    const void* bias = nullptr;

    float min_clip = -3.4e38f, max_clip = 3.4e38f;
    float ep_alpha = 0.0f, ep_beta = 1.0f;
    EpilogueActivateType ep = EpilogueActivateType::None;
    int add_to = 0, pre_act = 0;
};

/// The k index (tap·IC/G + ic) of the reduction, decoded once per k.
/// Conv3D's DHWIO filter is in this order; TransposeConv3D's is not, and walks
/// its own strides from the same three tap indices.
__device__ __forceinline__ void c3_decode(const C3Params& p, int k, int& kd, int& kh, int& kw,
                                          int& ic) noexcept
{
    const int tap = k / p.icg;
    ic = k - tap * p.icg;
    kd = tap / (p.kh * p.kw);
    const int r = tap - kd * p.kh * p.kw;
    kh = r / p.kw;
    kw = r - kh * p.kw;
}

/// The activation element a (voxel, k) pair names, or zero for a tap outside the
/// volume. `in` is already this batch·group's slice.
///
/// Conv3D walks forward: i = o·s − p + k·d. TransposeConv3D walks back:
/// o + p − k·d must divide by the stride, and the quotient is i. The remainder
/// test is correct for negative numerators too — C++ truncation makes an exact
/// divisible value's remainder zero either way — and the range test then drops
/// everything left of the volume.
template <bool TRANS, class Elem>
__device__ __forceinline__ float c3_gather(const C3Params& p, const Elem* in, int od, int oh,
                                           int ow, int kd, int kh, int kw, int ic) noexcept
{
    int64_t id, ih, iw;
    if constexpr (TRANS) {
        const int nd = od + p.pd - kd * p.dd;
        const int nh = oh + p.ph - kh * p.dh;
        const int nw = ow + p.pw - kw * p.dw;
        if (nd % p.sd != 0 || nh % p.sh != 0 || nw % p.sw != 0) {
            return 0.0f;
        }
        id = nd / p.sd;
        ih = nh / p.sh;
        iw = nw / p.sw;
    } else {
        id = int64_t(od) * p.sd - p.pd + kd * p.dd;
        ih = int64_t(oh) * p.sh - p.ph + kh * p.dh;
        iw = int64_t(ow) * p.sw - p.pw + kw * p.dw;
    }
    if (id < 0 || id >= p.id || ih < 0 || ih >= p.ih || iw < 0 || iw >= p.iw) {
        return 0.0f;
    }
    return s_load(in + id * p.in_d + ih * p.in_h + iw * p.in_w + ic * p.in_c);
}

/// Four panel elements as floats. The f32 panel is one 128-bit load; the 2-byte
/// panels are two pairs, which is one conversion each rather than four.
template <class Elem> struct C3Pair;
template <> struct C3Pair<float> { using type = float; };
template <> struct C3Pair<__half> { using type = __half2; };
template <> struct C3Pair<__nv_bfloat16> { using type = __nv_bfloat162; };

__device__ __forceinline__ float2 c3_pair_to_float2(__half2 v) noexcept
{
    return __half22float2(v);
}
__device__ __forceinline__ float2 c3_pair_to_float2(__nv_bfloat162 v) noexcept
{
    return __bfloat1622float2(v);
}

template <class Elem>
__device__ __forceinline__ void c3_ld4(const Elem* p, float (&o)[4]) noexcept
{
    if constexpr (sizeof(Elem) == 4) {
        const float4 v = *reinterpret_cast<const float4*>(p);
        o[0] = v.x;
        o[1] = v.y;
        o[2] = v.z;
        o[3] = v.w;
    } else {
        using P = typename C3Pair<Elem>::type;
        const float2 a = c3_pair_to_float2(*reinterpret_cast<const P*>(p));
        const float2 b = c3_pair_to_float2(*reinterpret_cast<const P*>(p + 2));
        o[0] = a.x;
        o[1] = a.y;
        o[2] = b.x;
        o[3] = b.y;
    }
}

template <class Elem, bool TRANS>
__global__ void __launch_bounds__(kC3Threads)
conv3d_kernel(C3Params p)
{
    __shared__ Elem sA[kC3BK * kC3SA];
    __shared__ Elem sB[kC3BK * kC3SB];
    __shared__ int s_od[kC3BM], s_oh[kC3BM], s_ow[kC3BM];
    __shared__ int s_kd[kC3BK], s_kh[kC3BK], s_kw[kC3BK], s_ic[kC3BK];

    const int tid = threadIdx.x;
    const int64_t m0 = int64_t(blockIdx.x) * kC3BM;
    const int64_t n0 = int64_t(blockIdx.y) * kC3BN;
    const int64_t g = int64_t(blockIdx.z) % p.groups;
    const int64_t b = int64_t(blockIdx.z) / p.groups;

    const int tm = tid >> 4;  // tile row index of this thread's first voxel
    const int tn = tid & 15;  // tile column index of this thread's first channel
    const int mv = tm * kC3TM;
    const int nv = tn * kC3TN;

    // The tile's voxel coordinates, shared because the fill walks rows the
    // compute does not own; one set of divisions a block, not one a element.
    if (tid < kC3BM) {
        const int64_t v = m0 + tid;
        if (v < p.m) {
            const int64_t ow = v % p.ow;
            const int64_t r = v / p.ow;
            s_ow[tid] = static_cast<int>(ow);
            s_oh[tid] = static_cast<int>(r % p.oh);
            s_od[tid] = static_cast<int>(r / p.oh);
        }
    }

    const Elem* in = static_cast<const Elem*>(p.in) + b * p.in_batch + g * p.icg * p.in_c;
    const Elem* w = static_cast<const Elem*>(p.w) + g * p.w_group;
    Elem* out = static_cast<Elem*>(p.out) + b * p.out_batch + g * p.ocg * p.out_c;
    const Elem* bias = p.bias != nullptr ? static_cast<const Elem*>(p.bias) + g * p.ocg : nullptr;

    const int64_t K = int64_t(p.icg) * p.kd * p.kh * p.kw;

    float acc[kC3TM][kC3TN];
#pragma unroll
    for (int i = 0; i < kC3TM; ++i) {
#pragma unroll
        for (int j = 0; j < kC3TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    for (int64_t k0 = 0; k0 < K; k0 += kC3BK) {
        const int64_t rem = K - k0;
        const int kn = static_cast<int>(rem < kC3BK ? rem : kC3BK);
        // Decode this chunk's taps. It races nothing the previous step reads,
        // and the barrier below both publishes it and closes that step's
        // panel reads.
        if (tid < kC3BK) {
            int kd = 0, kh = 0, kw = 0, ic = 0;
            if (tid < kn) {
                c3_decode(p, static_cast<int>(k0) + tid, kd, kh, kw, ic);
            }
            s_kd[tid] = kd;
            s_kh[tid] = kh;
            s_kw[tid] = kw;
            s_ic[tid] = ic;
        }
        __syncthreads();

        // A: the gathered activation, written k-major so the inner loop reads a
        // row of voxels. The fill is k-fastest for the same reason the other
        // way round: a warp then reads one tap's channel run.
        for (int idx = tid; idx < kC3BM * kC3BK; idx += kC3Threads) {
            const int mm = idx >> 5;
            const int c = idx & (kC3BK - 1);
            float v = 0.0f;
            if (c < kn && m0 + mm < p.m) {
                v = c3_gather<TRANS, Elem>(p, in, s_od[mm], s_oh[mm], s_ow[mm], s_kd[c], s_kh[c],
                                           s_kw[c], s_ic[c]);
            }
            s_store(sA + c * kC3SA + mm, v);
        }
        // B: the filter. n is the fast axis, which Conv3D's DHWIO filter has
        // contiguous; TransposeConv3D's walks a stride and is gathered instead.
        for (int idx = tid; idx < kC3BK * kC3BN; idx += kC3Threads) {
            const int c = idx >> 6;
            const int nn = idx & (kC3BN - 1);
            float v = 0.0f;
            if (c < kn && n0 + nn < p.ocg) {
                v = s_load(w + s_ic[c] * p.w_ic + (n0 + nn) * p.w_oc + s_kd[c] * p.w_kd +
                           s_kh[c] * p.w_kh + s_kw[c] * p.w_kw);
            }
            s_store(sB + c * kC3SB + nn, v);
        }
        __syncthreads();

#pragma unroll
        for (int c = 0; c < kC3BK; ++c) {
            float a[kC3TM], bb[kC3TN];
            c3_ld4(sA + c * kC3SA + mv, a);
            c3_ld4(sB + c * kC3SB + nv, bb);
#pragma unroll
            for (int i = 0; i < kC3TM; ++i) {
#pragma unroll
                for (int j = 0; j < kC3TN; ++j) {
                    acc[i][j] = fmaf(a[i], bb[j], acc[i][j]);
                }
            }
        }
    }

    // Write-back: bias, the residual in the order the operator asked for, the
    // clamp and the activation. The epilogue address is the operator's own, not
    // a strided C, because an NDHWC row is pitched.
#pragma unroll
    for (int i = 0; i < kC3TM; ++i) {
        const int64_t v = m0 + mv + i;
        if (v >= p.m) {
            continue;
        }
        const int64_t base = int64_t(s_od[mv + i]) * p.out_d + int64_t(s_oh[mv + i]) * p.out_h +
                             int64_t(s_ow[mv + i]) * p.out_w;
#pragma unroll
        for (int j = 0; j < kC3TN; ++j) {
            const int64_t n = n0 + nv + j;
            if (n >= p.ocg) {
                continue;
            }
            Elem* dst = out + base + n * p.out_c;
            float x = acc[i][j];
            if (bias != nullptr) {
                x += s_load(bias + n);
            }
            if (p.add_to && p.pre_act) {
                x += s_load(dst);
            }
            x = fminf(fmaxf(x, p.min_clip), p.max_clip);
            x = gemm_apply_activation(p.ep, p.ep_alpha, p.ep_beta, x);
            if (p.add_to && !p.pre_act) {
                x += s_load(dst);
            }
            s_store(dst, x);
        }
    }
}

template <class Elem, bool TRANS>
inline Status c3_launch(const C3Params& p, dim3 grid, cudaStream_t stream)
{
    conv3d_kernel<Elem, TRANS><<<grid, kC3Threads, 0, stream>>>(p);
    return cudaGetLastError() == cudaSuccess ? Status::ok : Status::internal;
}

} // namespace nnops::backend::cuda::conv3d
