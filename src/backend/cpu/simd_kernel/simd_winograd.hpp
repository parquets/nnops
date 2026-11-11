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
/// Layouts. Both orders appear in both layouts:
///
///   * Channel-first puts the *tile* axis in the vector lanes. F(2,3) reads
///     eight horizontally adjacent tiles from two stride-2 loads (offsets 0 and
///     2), matching NEON vld2q and compiling to the portable v_load_even /
///     v_load_odd primitives on both arches. Two caller obligations follow: the
///     8 tiles must lie in the same tile *row* (the tile grid cannot be
///     flattened to a 1-D index), and the padded halo row must be at least
///     2·tile_block + 2 wide — the offset-2 loads touch element
///     2·(tile_block−8) + 17. F(4,3) needs a stride-4 gather the portable layer
///     does not have and deinterleaves twice to build it.
///   * Channel-last puts the *channel* axis in the vector lanes: a window is
///     sixteen (F(2,3)) or thirty-six (F(4,3)) channel-vector loads and an
///     output tile four or sixteen stores, with no deinterleave and no
///     neighbour coupling, so its tile grid can flatten. The channel-first halo
///     is a padded-grid strip that must be zero outside the image; V's
///     transform position p lives at v_base + p·v_plane_stride, tile tw0+t at +t.
///
/// The output paths apply bias and epilogue on the write-back, per element: our
/// bias is per output *channel* (epilogue_inplace's is per *column*), and
/// because every element is written exactly once the residual add can read the
/// pre-existing output in place — no scratch buffer, no second pass.

#include "nnops/detail/simd/simd.hpp"
#include "nnops/core/epilogue.hpp"
#include "simd_epilogue.hpp"  // epilogue_inplace's scalar apply_epilogue

#include <cstddef>
#include <cstdint>

namespace nnops::kernel {

using namespace simd;

// ---- in-register widen / narrow shims ------------------------------------

inline v_f32x8 wino_widen(const v_f32x8& v) noexcept { return v; }
inline v_f32x8 wino_widen(const v_f16x8& v) noexcept { return v_cvt_f16_to_f32(v); }
inline void wino_store(float* p, const v_f32x8& v) noexcept { v_store(p, v); }
inline void wino_store(half* p, const v_f32x8& v) noexcept {
    v_store(p, v_cvt_f32_to_f16(v));
}

// =========================================================================
//  F(2×2, 3×3)
// =========================================================================

// ---- G g Gᵀ — 3×3 kernel → 4×4 transform-domain kernel --------------------

/// Transform one 3×3 kernel into its 4×4 Winograd form U = G g Gᵀ (scalar: prepack runs it once per (oc, ic)). Strides in elements; the caller stores position p = i·4 + j as ld_u = 4·plane, ld_c = plane.
template <class T>
inline void winograd_weight_transform(const T* g, int ld_g, T* u, int ld_u,
                                      int ld_c = 1) noexcept {
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

// ---- Bᵀ d B — input window → V -------------------------------------------

/// Transform 8 horizontally adjacent 4×4 input windows into V = Bᵀ d B.
template <class T>
inline void winograd_input_tile8(const T* halo, int halo_ld,
                                 int ph0, int pw0,
                                 T* v_base, int v_plane_stride) noexcept {
    // E = d B kept in an f32 stack buffer: all 16 live in registers would spill
    // on 16-ymm x86.
    alignas(32) float e[4][4][8];
    for (int r = 0; r < 4; ++r) {
        const T* row = halo + static_cast<ptrdiff_t>(ph0 + r) * halo_ld + pw0;
        // Two stride-2 loads (offsets 0 and 2): lane t of d_k is window column 2·t + k.
        const v_f32x8 d0 = wino_widen(v_load_even(row));
        const v_f32x8 d1 = wino_widen(v_load_odd(row));
        const v_f32x8 d2 = wino_widen(v_load_even(row + 2));
        const v_f32x8 d3 = wino_widen(v_load_odd(row + 2));
        v_store(&e[r][0][0], v_sub(d0, d2));
        v_store(&e[r][1][0], v_add(d1, d2));
        v_store(&e[r][2][0], v_sub(d2, d1));
        v_store(&e[r][3][0], v_sub(d1, d3));
    }
    for (int j = 0; j < 4; ++j) {
        const v_f32x8 e0 = v_load(&e[0][j][0]);
        const v_f32x8 e1 = v_load(&e[1][j][0]);
        const v_f32x8 e2 = v_load(&e[2][j][0]);
        const v_f32x8 e3 = v_load(&e[3][j][0]);
        wino_store(v_base + (0 * 4 + j) * v_plane_stride, v_sub(e0, e2));
        wino_store(v_base + (1 * 4 + j) * v_plane_stride, v_add(e1, e2));
        wino_store(v_base + (2 * 4 + j) * v_plane_stride, v_sub(e2, e1));
        wino_store(v_base + (3 * 4 + j) * v_plane_stride, v_sub(e1, e3));
    }
}

// ---- Aᵀ M A — M → 2×2 output tile ----------------------------------------

/// Transform 8 horizontally adjacent tiles' M into their 2×2 outputs, written straight into the output plane.
template <class T>
inline void winograd_output_tile8(const T* m_base, int m_plane_stride,
                                  int m_row_stride, int oc, int t0,
                                  T* out0, T* out1,
                                  int ow0, int ow_end, bool row1_ok,
                                  const Epilogue& ep, float bias_oc,
                                  bool add_to, bool pre_act = false) noexcept {
    // Aᵀ M per column c: the four M rows sit 4·m_plane_stride apart.
    v_f32x8 t[2][4];
    for (int c = 0; c < 4; ++c) {
        const T* mp = m_base + c * m_plane_stride
                           + static_cast<ptrdiff_t>(oc) * m_row_stride + t0;
        const v_f32x8 m0 = wino_widen(v_load(mp + 0 * 4 * m_plane_stride));
        const v_f32x8 m1 = wino_widen(v_load(mp + 1 * 4 * m_plane_stride));
        const v_f32x8 m2 = wino_widen(v_load(mp + 2 * 4 * m_plane_stride));
        const v_f32x8 m3 = wino_widen(v_load(mp + 3 * 4 * m_plane_stride));
        t[0][c] = v_add(v_add(m0, m1), m2);
        t[1][c] = v_sub(v_sub(m1, m2), m3);
    }
    // Columns 2t and 2t+1 come from separate vectors (no portable
    // v_interleave), so the write-back is scalar strided stores masked at
    // ow_end; row1_ok drops the second row when OH is odd.
    for (int r = 0; r < 2; ++r) {
        if (r == 1 && !row1_ok) { break; }
        const v_f32x8 o0 = v_add(v_add(t[r][0], t[r][1]), t[r][2]);
        const v_f32x8 o1 = v_sub(v_sub(t[r][1], t[r][2]), t[r][3]);
        alignas(32) float b0[8], b1[8];
        v_store(b0, o0);
        v_store(b1, o1);
        T* row = (r == 0) ? out0 : out1;
        for (int lane = 0; lane < 8; ++lane) {
            const int ow = ow0 + 2 * lane;
            if (ow >= ow_end) { break; }
            // pre_act is the ResNet order: the residual joins the sum before
            // the activation. The read is the same either way.
            backend::cpu::apply_epilogue_store<T>(row + ow, b0[lane] + bias_oc, ep,
                                                  oc, add_to, pre_act);
            if (ow + 1 < ow_end) {
                backend::cpu::apply_epilogue_store<T>(
                    row + ow + 1, b1[lane] + bias_oc, ep, oc, add_to, pre_act);
            }
        }
    }
}

// =========================================================================
//  F(2×2, 3×3) channel-last transforms
// =========================================================================

/// A zero vector in the transform domain (f32 whatever the storage type).
inline v_f32x8 wino_zero() noexcept { return v_zero(static_cast<const float*>(nullptr)); }

/// Load @p lanes (1..8) values at @p p into an f32 vector, zero above them —
/// the k tail of a channel block, which has no full vector to read.
template <class T>
inline v_f32x8 wino_load_n(const T* p, int lanes) noexcept {
    alignas(32) T buf[8] = {};
    for (int i = 0; i < lanes; ++i) { buf[i] = p[i]; }
    return wino_widen(v_load(buf));
}

template <class T>
inline v_f32x8 wino_load_lanes(const T* p, int lanes) noexcept {
    return (lanes >= simd_lane_for<T>) ? wino_widen(v_load(p)) : wino_load_n<T>(p, lanes);
}

/// Store the low @p lanes lanes of @p v at @p p, leaving the rest untouched.
template <class T>
inline void wino_store_lanes(T* p, const v_f32x8& v, int lanes) noexcept {
    if (lanes >= simd_lane_for<T>) { wino_store(p, v); return; }
    alignas(32) T buf[8];
    wino_store(buf, v);
    for (int i = 0; i < lanes; ++i) { p[i] = buf[i]; }
}

/// Transform one 4×4 input window into V = Bᵀ d B, one channel vector at a time.
template <class T>
inline void winograd_input_channel_last(const T* in_plane, int64_t in_row, int64_t ic_stride,
                                int ih0, int iw0, int IH, int IW,
                                T* v, int64_t v_plane, int lanes) noexcept {
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
            d[c] = (row != nullptr && w >= 0 && w < IW)
                 ? wino_load_lanes<T>(row + w * ic_stride, lanes)
                 : zero;
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
inline void winograd_output_channel_last(const T* m, int64_t m_plane,
                                 T* const p[2][2],
                                 const Epilogue& ep, const T* bias, bool add_to,
                                 int lanes, bool pre_act = false) noexcept {
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
    const float tag = 0.0f;  // epilogue dispatch only; the value is never read
    const auto emit = [&](T* dst, const v_f32x8& y) {
        if (dst == nullptr) { return; }
        v_f32x8 v = v_add(y, vb);
        if (add_to && pre_act) { v = v_add(v, wino_load_lanes<T>(dst, lanes)); }
        // apply_epilogue_vec is backend/cpu's (this header is a kernel's): same
        // file, same table, so the two forms cannot disagree.
        v = backend::cpu::apply_epilogue_vec<float>(ep, &tag, v);
        if (add_to && !pre_act) { v = v_add(v, wino_load_lanes<T>(dst, lanes)); }
        wino_store_lanes(dst, v, lanes);
    };
    for (int r = 0; r < 2; ++r) {
        if (p[r][0] == nullptr && p[r][1] == nullptr) { continue; }
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
constexpr int kWino43Pos   = kWino43Alpha * kWino43Alpha;
constexpr int kWino43Tile  = 4;

// ---- G g Gᵀ — 3×3 kernel → 6×6 transform-domain kernel -------------------

/// Transform one 3×3 kernel into its 6×6 Winograd form U = G g Gᵀ (scalar:
/// prepack runs it once per (oc, ic)). Strides in elements; the caller stores
/// position p = i·6 + j as ld_u = 6·plane, ld_c = plane.
template <class T>
inline void winograd43_weight_transform(const T* g, int ld_g, T* u, int ld_u,
                                        int ld_c = 1) noexcept {
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
inline void wino43_bt(v_f32x8 d[6]) noexcept {
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
                                  int ih0, int iw0, int IH, int IW,
                                  T* v, int64_t v_plane, int lanes) noexcept {
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
            d[c] = (row != nullptr && w >= 0 && w < IW)
                 ? wino_load_lanes<T>(row + w * ic_stride, lanes)
                 : zero;
        }
        wino43_bt(d);
        for (int c = 0; c < 6; ++c) { v_store(&e[r][c][0], d[c]); }
    }
    for (int j = 0; j < 6; ++j) {
        v_f32x8 d[6];
        for (int i = 0; i < 6; ++i) { d[i] = v_load(&e[i][j][0]); }
        wino43_bt(d);
        for (int i = 0; i < 6; ++i) {
            wino_store_lanes(v + (i * 6 + j) * v_plane, d[i], lanes);
        }
    }
}

// ---- Aᵀ M A — M → 4×4 output tile ----------------------------------------

/// Aᵀ applied to six f32 vectors (one M column, or one row of Aᵀ M).
inline void wino43_at(v_f32x8 m[6]) noexcept {
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
inline void winograd43_output_channel_last(const T* m, int64_t m_plane,
                                   T* const p[4][4],
                                   const Epilogue& ep, const T* bias, bool add_to,
                                   int lanes, bool pre_act = false) noexcept {
    // Aᵀ M per column, then (Aᵀ M) A.
    alignas(32) float t[4][6][8];
    for (int c = 0; c < 6; ++c) {
        v_f32x8 col[6];
        for (int i = 0; i < 6; ++i) { col[i] = wino_load_lanes<T>(m + c * m_plane + i * 6 * m_plane, lanes); }
        wino43_at(col);
        for (int r = 0; r < 4; ++r) { v_store(&t[r][c][0], col[r]); }
    }
    const v_f32x8 vb = (bias != nullptr) ? wino_load_lanes<T>(bias, lanes) : wino_zero();
    const float tag = 0.0f;  // epilogue dispatch only; the value is never read
    const auto emit = [&](T* dst, const v_f32x8& y) {
        if (dst == nullptr) { return; }
        v_f32x8 v = v_add(y, vb);
        if (add_to && pre_act) { v = v_add(v, wino_load_lanes<T>(dst, lanes)); }
        v = backend::cpu::apply_epilogue_vec<float>(ep, &tag, v);
        if (add_to && !pre_act) { v = v_add(v, wino_load_lanes<T>(dst, lanes)); }
        wino_store_lanes(dst, v, lanes);
    };
    for (int r = 0; r < 4; ++r) {
        v_f32x8 row[6];
        for (int c = 0; c < 6; ++c) { row[c] = v_load(&t[r][c][0]); }
        wino43_at(row);
        for (int c = 0; c < 4; ++c) { emit(p[r][c], row[c]); }
    }
}

// =========================================================================
//  F(4×4, 3×3) channel-first transforms — the tile axis is the vector lane
// =========================================================================
//
// Eight horizontally adjacent tiles sit four columns apart, and the portable
// SIMD layer only deinterleaves by two. Twice, though, is four: the even
// elements of the even elements are the elements four apart, so a stride-4
// gather is two v_load_even passes with the intermediate staged in a register
// pair. Each gather costs a store and a reload of that pair; six of them per
// window row is still cheaper than the multiplies they feed.

/// Gather @p p[0], @p p[4], … @p p[28] — eight tiles' column @c of one window
/// row. Reads p[0..31]; the caller's halo strip must own that span.
template <class T>
inline v_f32x8 wino43_stride4(const T* p) noexcept {
    alignas(32) T ev[16];
    wino_store(ev,     wino_widen(v_load_even(p)));       // p0, p2, … p14
    wino_store(ev + 8, wino_widen(v_load_even(p + 16)));  // p16, p18, … p30
    return wino_widen(v_load_even(ev));                   // p0, p4, … p28
}

/// Transform 8 horizontally adjacent 6×6 windows into V = Bᵀ d B. @p halo_ld
/// must own tile·tile_block + 8 columns from @p pw0 (the stride-4 gathers of
/// the last group reach four past its last tile).
template <class T>
inline void winograd43_input_tile8(const T* halo, int halo_ld,
                                   int ph0, int pw0,
                                   T* v_base, int v_plane_stride) noexcept {
    alignas(32) T rowbuf[40];        // the six window columns of eight tiles
    alignas(32) float e[6][6][8];
    for (int r = 0; r < 6; ++r) {
        const T* row = halo + static_cast<ptrdiff_t>(ph0 + r) * halo_ld + pw0;
        for (int i = 0; i < 40; ++i) { rowbuf[i] = row[i]; }
        v_f32x8 d[6];
        for (int k = 0; k < 6; ++k) { d[k] = wino43_stride4<T>(rowbuf + k); }
        wino43_bt(d);
        for (int k = 0; k < 6; ++k) { v_store(&e[r][k][0], d[k]); }
    }
    for (int j = 0; j < 6; ++j) {
        v_f32x8 d[6];
        for (int i = 0; i < 6; ++i) { d[i] = v_load(&e[i][j][0]); }
        wino43_bt(d);
        for (int i = 0; i < 6; ++i) {
            wino_store(v_base + (i * 6 + j) * v_plane_stride, d[i]);
        }
    }
}

/// Transform 8 horizontally adjacent tiles' 36 M planes into their 4×4
/// outputs, written straight into the output plane: @p out holds the first
/// @p rows_ok rows of the tile, and columns past @p ow_end are masked.
template <class T>
inline void winograd43_output_tile8(const T* m_base, int m_plane_stride,
                                    int m_row_stride, int oc, int t0,
                                    T* const out[4],
                                    int ow0, int ow_end, int rows_ok,
                                    const Epilogue& ep, float bias_oc,
                                    bool add_to, bool pre_act = false) noexcept {
    alignas(32) float t[4][6][8];
    for (int c = 0; c < 6; ++c) {
        const T* mp = m_base + c * m_plane_stride
                           + static_cast<ptrdiff_t>(oc) * m_row_stride + t0;
        v_f32x8 col[6];
        for (int i = 0; i < 6; ++i) {
            col[i] = wino_widen(v_load(mp + i * 6 * m_plane_stride));
        }
        wino43_at(col);
        for (int r = 0; r < 4; ++r) { v_store(&t[r][c][0], col[r]); }
    }
    for (int r = 0; r < rows_ok; ++r) {
        v_f32x8 row[6];
        for (int c = 0; c < 6; ++c) { row[c] = v_load(&t[r][c][0]); }
        wino43_at(row);
        for (int c = 0; c < 4; ++c) {
            alignas(32) float b[8];
            v_store(b, row[c]);
            for (int lane = 0; lane < 8; ++lane) {
                const int ow = ow0 + 4 * lane + c;
                if (ow >= ow_end) { break; }
                backend::cpu::apply_epilogue_store<T>(
                    out[r] + ow, b[lane] + bias_oc, ep, oc, add_to, pre_act);
            }
        }
    }
}

}  // namespace nnops::kernel
