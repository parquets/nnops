/// @file linear_attention_wy_host.hpp
/// @brief The chunked-WY route: a chunk-local panel kernel (the cumulative
///        decay, the gated Q/K panels, the triangular solve and W/U) and the
///        recurrent kernel that walks the panels in order.
///
/// The algebra is the CPU chunk route's, with the triangular solve already done
/// by the time the recurrence reads it. Per chunk c and (batch, value head):
///   X     = U - W S                    deltas
///   Aq    = Qe Ka^T                    gated intra-chunk product, s <= r
///   O     = scale (Qe S + tril(Aq) X)
///   S_new = Lc . (S + Ka^T X)
/// where Qe = q . L_r and Ka = k / L_s are the gate-scaled query and key panels
/// (L_r the exponential of the cumulative log decay to row r), and Lc is the
/// chunk's total decay, per key channel.
///
/// The operator drives `la_wy_route_cuda` over its raw inputs; the bench and the
/// tests drive `la_wy_recurrent_cuda` over synthesized panels.
///
/// The panel representation holds `1/L_s` in f16, so a chunk whose decay range
/// leaves the f16 exponent range cannot be represented — the panel kernel
/// reports whether its panels stayed inside it, and the route answers false so
/// the caller falls back to the per-token kernel. The bound is the physics of
/// the representation, the f16 analogue of the CPU route's kChunkSafeLog guard.

#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace nnops::backend::cuda {

/// The panels the recurrent kernel reads. Fixed geometry for the route: the
/// tensor-core chunk (C = 64) at Dk = Dv = 128.
struct LaWyParams {
    const void* w = nullptr;   ///< [B, Hv, L, C, Dk] f16
    const void* u = nullptr;   ///< [B, Hv, L, C, Dv] f16
    const void* qe = nullptr;  ///< [B, Hv, L, C, Dk] f16, q . L_r
    const void* ka = nullptr;  ///< [B, Hv, L, C, Dk] f16, k / L_s
    const void* lc = nullptr;  ///< [B, Hv, L, Dk] f32, the chunk's total decay
    void* o = nullptr;         ///< [B, Hv, S, Dv] f16
    void* st = nullptr;        ///< [B, Hv, Dk, Dv] f16, read then overwritten

    int64_t b = 0;
    int64_t hv = 0;
    int64_t s = 0;
    int64_t l = 0;
    float scale = 1.0f;

    /// Output and state strides, in elements, for a view that is not packed.
    /// 0 means packed: O at [B, Hv, S, Dv] and the state at [B, Hv, Dk, Dv].
    /// The last axis is assumed unit-stride, as the per-token kernel assumes.
    int64_t os0 = 0, os1 = 0, os2 = 0;
    int64_t ts0 = 0, ts1 = 0, ts2 = 0;
};

/// The operator's own inputs, for the fused route. f16 only; the geometry is
/// the same C = 64, Dk = Dv = 128, H_q == H_v (Gqq == 1), S a whole number of
/// chunks.
struct LaWyRoute {
    const void* q = nullptr;    ///< [B, Hv, S, Dk]
    const void* k = nullptr;    ///< [B, Hk, S, Dk]
    const void* v = nullptr;    ///< [B, Hv, S, Dv]
    const void* g = nullptr;    ///< [B, Hv, S, Dk] log-space decay
    const void* cw = nullptr;   ///< conv taps [channels, 1, ck]; null = none
    const void* beta = nullptr; ///< [B, Hv, S]; null = 1
    void* o = nullptr;          ///< [B, Hv, S, Dv]
    void* st = nullptr;         ///< [B, Hv, Dk, Dv]

    int64_t b = 0, hv = 0, hk = 0, s = 0, dk = 0, dv = 0;
    int64_t gvk = 1; ///< H_v / H_k
    int64_t ck = 0;  ///< conv taps; 0 means no conv

    int64_t qs0 = 0, qs1 = 0, qs2 = 0;
    int64_t ks0 = 0, ks1 = 0, ks2 = 0;
    int64_t vs0 = 0, vs1 = 0, vs2 = 0;
    int64_t gs0 = 0, gs1 = 0, gs2 = 0;
    int64_t os0 = 0, os1 = 0, os2 = 0;
    int64_t ts0 = 0, ts1 = 0, ts2 = 0, ts3 = 0;
    int64_t cws0 = 0, cws2 = 0;
    int64_t bs0 = 0, bs1 = 0, bs2 = 0;
    int64_t k_ch = 0, v_ch = 0; ///< conv channel bases of the packed Q | K | V blob
    float scale = 1.0f;
};

/// Run the recurrence over the panels. False when the call is outside the pose
/// (f16 only, Dk = Dv = 128, C = 64, S a whole number of chunks).
bool la_wy_recurrent_cuda(const LaWyParams& p, cudaStream_t stream);

/// Build the panels from the operator's inputs and run the recurrence. False
/// when the pose is unsupported or a chunk's decay range leaves the f16 panel
/// range; the caller then runs the per-token kernel instead. No output is
/// written on the false path (the panels are scratch).
///
/// The pose test blocks the caller's stream once, to learn whether the panels
/// stayed in range; the launch is otherwise asynchronous.
bool la_wy_route_cuda(const LaWyRoute& r, cudaStream_t stream);

/// How many calls the route has launched its recurrence for, over the process.
/// The operator's own call has no other way to say which route it took, so the
/// tests use this to prove the chunked route engaged rather than the fallback.
int64_t la_wy_route_runs();

} // namespace nnops::backend::cuda
