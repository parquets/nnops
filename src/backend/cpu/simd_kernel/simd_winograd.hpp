#pragma once
/// @file simd_winograd.hpp
/// @brief SIMD transforms for the two Winograd Conv2D paths (k3s1): F(2×2, 3×3)
///        and F(4×4, 3×3).
///
/// Both paths compute
///
///     Y = Aᵀ [ (G g Gᵀ) ⊙ (Bᵀ d B) ] A
///
/// in the *correlation* form (no kernel flip), matching the ONNX / nnops conv
/// convention, and both run every transform in f32 whatever the storage type:
/// an f16 round here would stack on top of the GEMM's f16 accumulate.
///
/// F(2×2, 3×3). One 2×2 output tile comes from a 4×4 input window and a 3×3
/// kernel: 36 multiplies for 16 (2.25× fewer), at the cost of the transforms
/// below. This section is the single source of truth for the three matrices. G
/// is read only by prepack (winograd_weight_transform), B and A only by the
/// kernels (winograd_input_* / winograd_output_*), so they cannot drift:
///
///   Bᵀ = [[ 1, 0,-1, 0],     G = [[ 1,   0,   0  ],     Aᵀ = [[1, 1, 1, 0],
///         [ 0, 1, 1, 0],           [ 1/2, 1/2, 1/2],           [0, 1,-1,-1]]
///         [ 0,-1, 1, 0],           [ 1/2,-1/2, 1/2],
///         [ 0, 1, 0,-1]]           [ 0,   0,   1  ]]
///
/// B and A carry only 0/±1, so those transforms are adds/subs (no multiply, no
/// FMA); G's 1/2s are applied once per (oc, ic) at prepack.
///
/// F(4×4, 3×3). One 4×4 output tile comes from a 6×6 input window: 36
/// multiplies for 144 outputs against F(2,3)'s 16 for 16 — 2.25 against 4
/// multiplies per output, so the GEMM term is 1.78x smaller. The transforms are
/// the price: B and A stop being 0/±1 and pick up 2, 4, 5 and 8, so an input
/// transform costs roughly 4.5x the F(2,3) one *per tile* and only ~1.1x per
/// output. That ratio is why this order needs more channels, not fewer, to pay.
/// For F(4,3) f32 is not just tidiness — Bᵀ scales a window by up to 5 before
/// the taps are combined and Aᵀ by up to 8 after — so f16 storage is allowed
/// but its V and U planes are the only f16 steps.
///
///   Bᵀ = [[ 4, 0,-5, 0, 1, 0],   G = [[ 1/4,    0,     0   ],   Aᵀ = [[1, 1, 1, 1,  1, 0],
///         [ 0,-4,-4, 1, 1, 0],        [-1/6, -1/6,  -1/6 ],         [0, 1,-1, 2, -2, 0],
///         [ 0, 4,-4,-1, 1, 0],        [-1/6,  1/6,  -1/6 ],         [0, 1, 1, 4,  4, 0],
///         [ 0,-2,-1, 2, 1, 0],        [ 1/24, 1/12,  1/6 ],         [0, 1,-1, 8, -8, 1]]
///         [ 0, 2,-1,-2, 1, 0],        [ 1/24,-1/12,  1/6 ],
///         [ 0, 4, 0,-5, 0, 1]]        [ 0,    0,     1   ]]
///
/// Layouts. Channel-last puts the *channel* axis in the vector lanes: a window
/// is sixteen (F(2,3)) or thirty-six (F(4,3)) channel-vector loads and an output
/// tile four or sixteen stores, with no deinterleave and no neighbour coupling,
/// so its tile grid can flatten. V's transform position p lives at
/// v_base + p·v_plane_stride, tile tw0+t at +t.
///
/// The output paths apply bias and epilogue on the write-back, per element: our
/// bias is per output *channel* (epilogue_inplace's is per *column*), and
/// because every element is written exactly once the residual add can read the
/// pre-existing output in place — no scratch buffer, no second pass.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/core/epilogue.hpp"
#include "simd_epilogue.hpp" // epilogue_inplace's scalar apply_epilogue

#include <cstddef>
#include <cstdint>

namespace nnops::kernel {

using namespace simd;

// ---- in-register widen / narrow shims ------------------------------------

inline v_f32x8 wino_widen(const v_f32x8& v) noexcept { return v; }
inline v_f32x8 wino_widen(const v_f16x8& v) noexcept { return v_cvt_f16_to_f32(v); }
inline void wino_store(float* p, const v_f32x8& v) noexcept { v_store(p, v); }
inline void wino_store(half* p, const v_f32x8& v) noexcept { v_store(p, v_cvt_f32_to_f16(v)); }

// =========================================================================
//  F(2×2, 3×3)
// =========================================================================

// ---- G g Gᵀ — 3×3 kernel → 4×4 transform-domain kernel --------------------

/// Transform one 3×3 kernel into its 4×4 Winograd form U = G g Gᵀ (scalar: prepack runs it once per (oc, ic)). Strides in elements; the caller stores position p = i·4 + j as ld_u = 4·plane, ld_c = plane.
template <class T>
inline void winograd_weight_transform(const T* g, int ld_g, T* u, int ld_u, int ld_c = 1) noexcept
{
    // G g, one column at a time so the whole 3×3 never needs to be live.
    float t[4][3];
    for (int j = 0; j < 3; ++j) {
        const float g0 = s_load(g + 0 * ld_g + j);
        const float g1 = s_load(g + 1 * ld_g + j);
        const float g2 = s_load(g + 2 * ld_g + j);
        t[0][j] = g0;
        t[1][j] = 0.5f * (g0 + g1 + g2);
        t[2][j] = 0.5f * (g0 - g1 + g2);
        t[3][j] = g2;
    }
    for (int i = 0; i < 4; ++i) {
        const float c0 = t[i][0], c1 = t[i][1], c2 = t[i][2];
        T* row = u + i * ld_u;
        s_store(row + 0 * ld_c, c0);
        s_store(row + 1 * ld_c, 0.5f * (c0 + c1 + c2));
        s_store(row + 2 * ld_c, 0.5f * (c0 - c1 + c2));
        s_store(row + 3 * ld_c, c2);
    }
}

// =========================================================================
//  F(2×2, 3×3) channel-last transforms
// =========================================================================

/// A zero vector in the transform domain (f32 whatever the storage type).
inline v_f32x8 wino_zero() noexcept { return v_zero(static_cast<const float*>(nullptr)); }

/// Load @p lanes (1..8) values at @p p into an f32 vector, zero above them —
/// the k tail of a channel block, which has no full vector to read.
template <class T> inline v_f32x8 wino_load_n(const T* p, int lanes) noexcept
{
    alignas(32) T buf[8] = {};
    for (int i = 0; i < lanes; ++i) {
        buf[i] = p[i];
    }
    return wino_widen(v_load(buf));
}

template <class T> inline v_f32x8 wino_load_lanes(const T* p, int lanes) noexcept
{
    return (lanes >= simd_lane_for<T>) ? wino_widen(v_load(p)) : wino_load_n<T>(p, lanes);
}

/// Store the low @p lanes lanes of @p v at @p p, leaving the rest untouched.
template <class T> inline void wino_store_lanes(T* p, const v_f32x8& v, int lanes) noexcept
{
    if (lanes >= simd_lane_for<T>) {
        wino_store(p, v);
        return;
    }
    alignas(32) T buf[8];
    wino_store(buf, v);
    for (int i = 0; i < lanes; ++i) {
        p[i] = buf[i];
    }
}

/// Transform one 4×4 input window into V = Bᵀ d B, one channel vector at a time.
template <class T>
inline void winograd_input_channel_last(const T* in_plane, int64_t in_row, int64_t ic_stride,
                                        int ih0, int iw0, int IH, int IW, T* v, int64_t v_plane,
                                        int lanes) noexcept
{
    // E = d B in an f32 stack buffer (16 live vectors would spill on 16-ymm x86);
    // the window is read row by row, so only four d vectors are ever live.
    alignas(32) float e[4][4][8];
    const v_f32x8 zero = wino_zero();
    for (int r = 0; r < 4; ++r) {
        const int h = ih0 + r;
        const T* row = (h >= 0 && h < IH) ? in_plane + h * in_row : nullptr;
        v_f32x8 d[4];
        for (int c = 0; c < 4; ++c) {
            const int w = iw0 + c;
            d[c] = (row != nullptr && w >= 0 && w < IW) ?
                       wino_load_lanes<T>(row + w * ic_stride, lanes) :
                       zero;
        }
        v_store(&e[r][0][0], v_sub(d[0], d[2]));
        v_store(&e[r][1][0], v_add(d[1], d[2]));
        v_store(&e[r][2][0], v_sub(d[2], d[1]));
        v_store(&e[r][3][0], v_sub(d[1], d[3]));
    }
    for (int j = 0; j < 4; ++j) {
        const v_f32x8 e0 = v_load(&e[0][j][0]);
        const v_f32x8 e1 = v_load(&e[1][j][0]);
        const v_f32x8 e2 = v_load(&e[2][j][0]);
        const v_f32x8 e3 = v_load(&e[3][j][0]);
        wino_store_lanes(v + (0 * 4 + j) * v_plane, v_sub(e0, e2), lanes);
        wino_store_lanes(v + (1 * 4 + j) * v_plane, v_add(e1, e2), lanes);
        wino_store_lanes(v + (2 * 4 + j) * v_plane, v_sub(e2, e1), lanes);
        wino_store_lanes(v + (3 * 4 + j) * v_plane, v_sub(e1, e3), lanes);
    }
}

/// Transform one tile's 16 M planes into its 2×2 outputs, one channel vector at a time; a nullptr destination skips that pixel (past OW/OH on an odd extent). @p p is the tile's destination grid, row-major.
template <class T>
inline void winograd_output_channel_last(const T* m, int64_t m_plane, T* const p[2][2],
                                         const Epilogue& ep, const T* bias, bool add_to, int lanes,
                                         bool pre_act = false) noexcept
{
    // Aᵀ M per column, then (Aᵀ M) A.
    alignas(32) float t[2][4][8];
    for (int c = 0; c < 4; ++c) {
        const T* mp = m + c * m_plane;
        const v_f32x8 m0 = wino_load_lanes<T>(mp + 0 * 4 * m_plane, lanes);
        const v_f32x8 m1 = wino_load_lanes<T>(mp + 1 * 4 * m_plane, lanes);
        const v_f32x8 m2 = wino_load_lanes<T>(mp + 2 * 4 * m_plane, lanes);
        const v_f32x8 m3 = wino_load_lanes<T>(mp + 3 * 4 * m_plane, lanes);
        v_store(&t[0][c][0], v_add(v_add(m0, m1), m2));
        v_store(&t[1][c][0], v_sub(v_sub(m1, m2), m3));
    }
    const v_f32x8 vb = (bias != nullptr) ? wino_load_lanes<T>(bias, lanes) : wino_zero();
    const float tag = 0.0f; // epilogue dispatch only; the value is never read
    const auto emit = [&](T* dst, const v_f32x8& y) {
        if (dst == nullptr) {
            return;
        }
        v_f32x8 v = v_add(y, vb);
        if (add_to && pre_act) {
            v = v_add(v, wino_load_lanes<T>(dst, lanes));
        }
        // apply_epilogue_vec is backend/cpu's (this header is a kernel's): same
        // file, same table, so the two forms cannot disagree.
        v = backend::cpu::apply_epilogue_vec<float>(ep, &tag, v);
        if (add_to && !pre_act) {
            v = v_add(v, wino_load_lanes<T>(dst, lanes));
        }
        wino_store_lanes(dst, v, lanes);
    };
    for (int r = 0; r < 2; ++r) {
        if (p[r][0] == nullptr && p[r][1] == nullptr) {
            continue;
        }
        const v_f32x8 t0 = v_load(&t[r][0][0]);
        const v_f32x8 t1 = v_load(&t[r][1][0]);
        const v_f32x8 t2 = v_load(&t[r][2][0]);
        const v_f32x8 t3 = v_load(&t[r][3][0]);
        emit(p[r][0], v_add(v_add(t0, t1), t2));
        emit(p[r][1], v_sub(v_sub(t1, t2), t3));
    }
}

// =========================================================================
//  F(4×4, 3×3)
// =========================================================================

/// The 6×6 transform domain of F(4,3): 36 batched GEMMs, a 4×4 output per tile.
constexpr int kWino43Alpha = 6;
constexpr int kWino43Pos = kWino43Alpha * kWino43Alpha;
constexpr int kWino43Tile = 4;

// ---- G g Gᵀ — 3×3 kernel → 6×6 transform-domain kernel -------------------

/// Transform one 3×3 kernel into its 6×6 Winograd form U = G g Gᵀ (scalar:
/// prepack runs it once per (oc, ic)). Strides in elements; the caller stores
/// position p = i·6 + j as ld_u = 6·plane, ld_c = plane.
template <class T>
inline void winograd43_weight_transform(const T* g, int ld_g, T* u, int ld_u, int ld_c = 1) noexcept
{
    // G g, one column at a time so the whole 3×3 never needs to be live.
    float t[6][3];
    for (int j = 0; j < 3; ++j) {
        const float g0 = s_load(g + 0 * ld_g + j);
        const float g1 = s_load(g + 1 * ld_g + j);
        const float g2 = s_load(g + 2 * ld_g + j);
        t[0][j] = 0.25f * g0;
        t[1][j] = -(g0 + g1 + g2) * (1.0f / 6.0f);
        t[2][j] = (g1 - g0 - g2) * (1.0f / 6.0f);
        t[3][j] = (g0 + 2.0f * g1 + 4.0f * g2) * (1.0f / 24.0f);
        t[4][j] = (g0 - 2.0f * g1 + 4.0f * g2) * (1.0f / 24.0f);
        t[5][j] = g2;
    }
    for (int i = 0; i < 6; ++i) {
        const float c0 = t[i][0], c1 = t[i][1], c2 = t[i][2];
        T* row = u + i * ld_u;
        s_store(row + 0 * ld_c, 0.25f * c0);
        s_store(row + 1 * ld_c, -(c0 + c1 + c2) * (1.0f / 6.0f));
        s_store(row + 2 * ld_c, (c1 - c0 - c2) * (1.0f / 6.0f));
        s_store(row + 3 * ld_c, (c0 + 2.0f * c1 + 4.0f * c2) * (1.0f / 24.0f));
        s_store(row + 4 * ld_c, (c0 - 2.0f * c1 + 4.0f * c2) * (1.0f / 24.0f));
        s_store(row + 5 * ld_c, c2);
    }
}

// ---- Bᵀ d B — input window → V -------------------------------------------

/// Bᵀ applied to six f32 vectors (one window row, or one column of the
/// half-transformed window). Kept as expressions rather than a matrix product:
/// every 0 term is dropped and the shared sums are formed once.
inline void wino43_bt(v_f32x8 d[6]) noexcept
{
    const v_f32x8 d0 = d[0], d1 = d[1], d2 = d[2], d3 = d[3], d4 = d[4], d5 = d[5];
    const v_f32x8 c2 = v_set1_f32x8(2.0f);
    const v_f32x8 c4 = v_set1_f32x8(4.0f);
    const v_f32x8 c5 = v_set1_f32x8(5.0f);
    const v_f32x8 s12 = v_add(d1, d2);
    const v_f32x8 m12 = v_sub(d1, d2);
    d[0] = v_add(v_sub(v_mul(d0, c4), v_mul(d2, c5)), d4);
    d[1] = v_add(v_sub(d3, v_mul(s12, c4)), d4);
    d[2] = v_add(v_sub(v_mul(m12, c4), d3), d4);
    d[3] = v_add(v_sub(v_mul(d3, c2), d2), v_sub(d4, v_mul(d1, c2)));
    d[4] = v_add(v_sub(v_mul(d1, c2), d2), v_sub(d4, v_mul(d3, c2)));
    d[5] = v_add(v_sub(v_mul(d1, c4), v_mul(d3, c5)), d5);
}

// =========================================================================
//  F(4×4, 3×3) channel-last transforms — the channel axis is the vector lane
// =========================================================================

/// Transform one 6×6 input window into V = Bᵀ d B, one channel vector at a time.
template <class T>
inline void winograd43_input_channel_last(const T* in_plane, int64_t in_row, int64_t ic_stride,
                                          int ih0, int iw0, int IH, int IW, T* v, int64_t v_plane,
                                          int lanes) noexcept
{
    // E = d B in an f32 stack buffer: 36 live vectors would spill every register
    // on both arches, and the window is read row by row, so only six d vectors
    // are ever live.
    alignas(32) float e[6][6][8];
    const v_f32x8 zero = wino_zero();
    for (int r = 0; r < 6; ++r) {
        const int h = ih0 + r;
        const T* row = (h >= 0 && h < IH) ? in_plane + h * in_row : nullptr;
        v_f32x8 d[6];
        for (int c = 0; c < 6; ++c) {
            const int w = iw0 + c;
            d[c] = (row != nullptr && w >= 0 && w < IW) ?
                       wino_load_lanes<T>(row + w * ic_stride, lanes) :
                       zero;
        }
        wino43_bt(d);
        for (int c = 0; c < 6; ++c) {
            v_store(&e[r][c][0], d[c]);
        }
    }
    for (int j = 0; j < 6; ++j) {
        v_f32x8 d[6];
        for (int i = 0; i < 6; ++i) {
            d[i] = v_load(&e[i][j][0]);
        }
        wino43_bt(d);
        for (int i = 0; i < 6; ++i) {
            wino_store_lanes(v + (i * 6 + j) * v_plane, d[i], lanes);
        }
    }
}

// ---- Aᵀ M A — M → 4×4 output tile ----------------------------------------

/// Aᵀ applied to six f32 vectors (one M column, or one row of Aᵀ M).
inline void wino43_at(v_f32x8 m[6]) noexcept
{
    const v_f32x8 m0 = m[0], m1 = m[1], m2 = m[2], m3 = m[3], m4 = m[4], m5 = m[5];
    const v_f32x8 s12 = v_add(m1, m2);
    const v_f32x8 m12 = v_sub(m1, m2);
    const v_f32x8 s34 = v_add(m3, m4);
    const v_f32x8 d34 = v_sub(m3, m4);
    m[0] = v_add(v_add(m0, s12), s34);
    m[1] = v_add(m12, v_mul(d34, v_set1_f32x8(2.0f)));
    m[2] = v_add(s12, v_mul(s34, v_set1_f32x8(4.0f)));
    m[3] = v_add(v_add(m12, v_mul(d34, v_set1_f32x8(8.0f))), m5);
}

/// Transform one tile's 36 M planes into its 4×4 outputs, one channel vector at
/// a time; a nullptr destination skips that pixel (past OW/OH on a short extent).
template <class T>
inline void winograd43_output_channel_last(const T* m, int64_t m_plane, T* const p[4][4],
                                           const Epilogue& ep, const T* bias, bool add_to,
                                           int lanes, bool pre_act = false) noexcept
{
    // Aᵀ M per column, then (Aᵀ M) A.
    alignas(32) float t[4][6][8];
    for (int c = 0; c < 6; ++c) {
        v_f32x8 col[6];
        for (int i = 0; i < 6; ++i) {
            col[i] = wino_load_lanes<T>(m + c * m_plane + i * 6 * m_plane, lanes);
        }
        wino43_at(col);
        for (int r = 0; r < 4; ++r) {
            v_store(&t[r][c][0], col[r]);
        }
    }
    const v_f32x8 vb = (bias != nullptr) ? wino_load_lanes<T>(bias, lanes) : wino_zero();
    const float tag = 0.0f; // epilogue dispatch only; the value is never read
    const auto emit = [&](T* dst, const v_f32x8& y) {
        if (dst == nullptr) {
            return;
        }
        v_f32x8 v = v_add(y, vb);
        if (add_to && pre_act) {
            v = v_add(v, wino_load_lanes<T>(dst, lanes));
        }
        v = backend::cpu::apply_epilogue_vec<float>(ep, &tag, v);
        if (add_to && !pre_act) {
            v = v_add(v, wino_load_lanes<T>(dst, lanes));
        }
        wino_store_lanes(dst, v, lanes);
    };
    for (int r = 0; r < 4; ++r) {
        v_f32x8 row[6];
        for (int c = 0; c < 6; ++c) {
            row[c] = v_load(&t[r][c][0]);
        }
        wino43_at(row);
        for (int c = 0; c < 4; ++c) {
            emit(p[r][c], row[c]);
        }
    }
}

} // namespace nnops::kernel
