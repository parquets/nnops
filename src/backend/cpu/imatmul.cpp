#include "imatmul.h"

#include <array>

#ifdef NNOPS_ARCH_X86_64
using namespace nnops::backend::cpu::x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
using namespace nnops::backend::cpu::aarch64;
#endif

namespace nnops::backend::cpu {

// ---- pack function pointer types ---------------------------------------

using PackF32Fn = void (*)(float* NNOPS_RESTRICT output,
                           const float* NNOPS_RESTRICT input,
                           int ir_step, int K, float scale);

using PackF16Fn = void (*)(half* NNOPS_RESTRICT output,
                           const half* NNOPS_RESTRICT input,
                           int ir_step, int K, float scale);

// ---- MMA-pack function pointer types -----------------------------------

using MmaPackF32Fn = void (*)(float* NNOPS_RESTRICT C, int ldc,
                              const float* NNOPS_RESTRICT A,
                              const float* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max);

using MmaPackF16Fn = void (*)(half* NNOPS_RESTRICT C, int ldc,
                              const half* NNOPS_RESTRICT A,
                              const half* NNOPS_RESTRICT B,
                              int ldb, int K,
                              float clamp_min, float clamp_max);

// ---- MMA-direct function pointer types ---------------------------------

using MmaDirectF32Fn = void (*)(float* NNOPS_RESTRICT C, int ldc,
                                const float* NNOPS_RESTRICT A, int lda,
                                const float* NNOPS_RESTRICT B, int ldb,
                                int K, float clamp_min, float clamp_max);

using MmaDirectF16Fn = void (*)(half* NNOPS_RESTRICT C, int ldc,
                                const half* NNOPS_RESTRICT A, int lda,
                                const half* NNOPS_RESTRICT B, int ldb,
                                int K, float clamp_min, float clamp_max);

// ---- generic panel-size → index lookup ---------------------------------

/// @brief Find the index of `size` in a compile-time panel-size array.
/// Returns the index if found, otherwise the last index (smallest panel).
template <size_t N>
constexpr int panel_index(int size, const int (&panel)[N]) noexcept {
    for (size_t i = 0; i < N; ++i) {
        if (size == panel[i]) { return static_cast<int>(i); }
    }
    return static_cast<int>(N) - 1;
}

// ---- pack dispatch tables (largest-first decomposition) -----------------

// f32 dispatch tables
constexpr std::array<PackF32Fn, 3> pack_trans_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    pack_trans_n6_f32,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_trans_n8_f32,
#endif
    pack_trans_n4_f32,
    pack_trans_n1_f32
}};

constexpr std::array<PackF32Fn, 3> pack_copy_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    pack_copy_n16_f32,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_copy_n12_f32,
#endif
    pack_copy_n8_f32,
    pack_copy_n1_f32
}};

// f16 dispatch tables
constexpr std::array<PackF16Fn, 3> pack_trans_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    pack_trans_n6_f16,
#elif defined(NNOPS_ARCH_AARCH64)
    pack_trans_n8_f16,
#endif
    pack_trans_n4_f16,
    pack_trans_n1_f16
}};

// pack_copy has identical decomposition on both x86_64 and aarch64
// (nr_f16 = {16, 8, 1}), so no #ifdef needed.
constexpr std::array<PackF16Fn, 3> pack_copy_f16_fn = {{
    pack_copy_n16_f16,
    pack_copy_n8_f16,
    pack_copy_n1_f16
}};

// ---- MMA dispatch tables -----------------------------------------------

constexpr std::array<std::array<MmaPackF32Fn, 3>, 3> mma_pack_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_pack_6x16_f32, mma_pack_6x8_f32, mma_pack_6x1_f32}},
    {{mma_pack_4x16_f32, mma_pack_4x8_f32, mma_pack_4x1_f32}},
    {{mma_pack_1x16_f32, mma_pack_1x8_f32, mma_pack_1x1_f32}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_pack_8x12_f32, mma_pack_8x4_f32, mma_pack_8x1_f32}},
    {{mma_pack_4x12_f32, mma_pack_4x4_f32, mma_pack_4x1_f32}},
    {{mma_pack_1x12_f32, mma_pack_1x4_f32, mma_pack_1x1_f32}},
#endif
}};

constexpr std::array<std::array<MmaPackF16Fn, 3>, 3> mma_pack_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_pack_6x16_f16, mma_pack_6x8_f16, mma_pack_6x1_f16}},
    {{mma_pack_4x16_f16, mma_pack_4x8_f16, mma_pack_4x1_f16}},
    {{mma_pack_1x16_f16, mma_pack_1x8_f16, mma_pack_1x1_f16}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_pack_8x16_f16, mma_pack_8x8_f16, mma_pack_8x1_f16}},
    {{mma_pack_4x16_f16, mma_pack_4x8_f16, mma_pack_4x1_f16}},
    {{mma_pack_1x16_f16, mma_pack_1x8_f16, mma_pack_1x1_f16}},
#endif
}};

constexpr std::array<std::array<MmaDirectF32Fn, 3>, 3> mma_direct_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_direct_6x16_f32, mma_direct_6x8_f32, mma_direct_6x1_f32}},
    {{mma_direct_4x16_f32, mma_direct_4x8_f32, mma_direct_4x1_f32}},
    {{mma_direct_1x16_f32, mma_direct_1x8_f32, mma_direct_1x1_f32}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_direct_8x12_f32, mma_direct_8x4_f32, mma_direct_8x1_f32}},
    {{mma_direct_4x12_f32, mma_direct_4x4_f32, mma_direct_4x1_f32}},
    {{mma_direct_1x12_f32, mma_direct_1x4_f32, mma_direct_1x1_f32}},
#endif
}};

constexpr std::array<std::array<MmaDirectF16Fn, 3>, 3> mma_direct_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {{mma_direct_6x16_f16, mma_direct_6x8_f16, mma_direct_6x1_f16}},
    {{mma_direct_4x16_f16, mma_direct_4x8_f16, mma_direct_4x1_f16}},
    {{mma_direct_1x16_f16, mma_direct_1x8_f16, mma_direct_1x1_f16}},
#elif defined(NNOPS_ARCH_AARCH64)
    {{mma_direct_8x16_f16, mma_direct_8x8_f16, mma_direct_8x1_f16}},
    {{mma_direct_4x16_f16, mma_direct_4x8_f16, mma_direct_4x1_f16}},
    {{mma_direct_1x16_f16, mma_direct_1x8_f16, mma_direct_1x1_f16}},
#endif
}};

// ---- slice pack functions (single micro-panel) -------------------------

// f32
void panel_pack_rhs(bool trans, int mr, int kc,
                  float* dst, const float* src, int lds, float scale) {
    if (trans) {
        pack_copy_f32_fn[panel_index(mr, MR_F32)](dst, src, lds, kc, scale);
    } else {
        pack_trans_f32_fn[panel_index(mr, NR_F32)](dst, src, lds, kc, scale);
    }
}

void panel_pack_lhs(bool trans, int nr, int kc,
                  float* dst, const float* src, int lds, float scale) {
    if (trans) {
        pack_trans_f32_fn[panel_index(nr, NR_F32)](dst, src, lds, kc, scale);
    } else {
        pack_copy_f32_fn[panel_index(nr, MR_F32)](dst, src, lds, kc, scale);
    }
}

// f16
void panel_pack_rhs(bool trans, int mr, int kc,
                  half* dst, const half* src, int lds, float scale) {
    if (trans) {
        pack_copy_f16_fn[panel_index(mr, MR_F16)](dst, src, lds, kc, scale);
    } else {
        pack_trans_f16_fn[panel_index(mr, NR_F16)](dst, src, lds, kc, scale);
    }
}

void panel_pack_lhs(bool trans, int nr, int kc,
                  half* dst, const half* src, int lds, float scale) {
    if (trans) {
        pack_trans_f16_fn[panel_index(nr, NR_F16)](dst, src, lds, kc, scale);
    } else {
        pack_copy_f16_fn[panel_index(nr, MR_F16)](dst, src, lds, kc, scale);
    }
}

// ---- tiled pack helpers ------------------------------------------------

namespace {

template <size_t N>
void tile_pack_rhs_loop(bool trans, int mc, int kc,
                        float* dst, const float* src, int lds, float scale,
                        const int (&panel)[N]) {
    int m = 0;
    for (size_t i = 0; i < N; ++i) {
        for (; m + panel[i] <= mc; m += panel[i]) {
            panel_pack_rhs(trans, panel[i], kc, dst + m * kc,
                         src + (trans ? m : m * lds), lds, scale);
        }
    }
}

template <size_t N>
void tile_pack_lhs_loop(bool trans, int nc, int kc,
                        float* dst, const float* src, int lds, float scale,
                        const int (&panel)[N]) {
    int n = 0;
    for (size_t i = 0; i < N; ++i) {
        for (; n + panel[i] <= nc; n += panel[i]) {
            panel_pack_lhs(trans, panel[i], kc, dst + n * kc,
                         src + (trans ? n * lds : n), lds, scale);
        }
    }
}

// ---- f16 tile pack helpers (half* overloads) -------------------------------

template <size_t N>
void tile_pack_rhs_loop(bool trans, int mc, int kc,
                        half* dst, const half* src, int lds, float scale,
                        const int (&panel)[N]) {
    int m = 0;
    for (size_t i = 0; i < N; ++i) {
        for (; m + panel[i] <= mc; m += panel[i]) {
            panel_pack_rhs(trans, panel[i], kc, dst + m * kc,
                         src + (trans ? m : m * lds), lds, scale);
        }
    }
}

template <size_t N>
void tile_pack_lhs_loop(bool trans, int nc, int kc,
                        half* dst, const half* src, int lds, float scale,
                        const int (&panel)[N]) {
    int n = 0;
    for (size_t i = 0; i < N; ++i) {
        for (; n + panel[i] <= nc; n += panel[i]) {
            panel_pack_lhs(trans, panel[i], kc, dst + n * kc,
                         src + (trans ? n * lds : n), lds, scale);
        }
    }
}

}  // anonymous namespace

// ---- tiled pack entry points -------------------------------------------

// f32
void tile_pack_rhs(bool trans, int mc, int kc,
                 float* dst, const float* src, int lds, float scale) {
    tile_pack_rhs_loop(trans, mc, kc, dst, src, lds, scale, MR_F32);
}

void tile_pack_lhs(bool trans, int nc, int kc,
                 float* dst, const float* src, int lds, float scale) {
    tile_pack_lhs_loop(trans, nc, kc, dst, src, lds, scale, NR_F32);
}

// f16
void tile_pack_rhs(bool trans, int mc, int kc,
                 half* dst, const half* src, int lds, float scale) {
    tile_pack_rhs_loop(trans, mc, kc, dst, src, lds, scale, MR_F16);
}

void tile_pack_lhs(bool trans, int nc, int kc,
                 half* dst, const half* src, int lds, float scale) {
    tile_pack_lhs_loop(trans, nc, kc, dst, src, lds, scale, NR_F16);
}

// ---- mma entry points ---------------------------------------------------

// f32 — pack
void panel_mma_pack(int Mr, int Nc, int Kc,
                  float* c, int ldc,
                  const float* a, const float* b, int ldb,
                  float clamp_min, float clamp_max) {
    const int mi = panel_index(Mr, MR_F32);
    const int ni = panel_index(Nc, NR_F32);
    mma_pack_f32_fn[mi][ni](c, ldc, a, b, ldb, Kc, clamp_min, clamp_max);
}

void tile_mma_pack(int Mc, int Nc, int Kc,
                 float* c, int ldc,
                 const float* a, const float* b, int ldb,
                 float clamp_min, float clamp_max) {
    // TODO: tiled loop over Mc/Nc using mma_pack_f32_fn panels
}

// f32 — direct
void panel_mma_direct(int Mr, int Nc, int Kc,
                    float* c, int ldc,
                    const float* a, int lda,
                    const float* b, int ldb,
                    float clamp_min, float clamp_max) {
    const int mi = panel_index(Mr, MR_F32);
    const int ni = panel_index(Nc, NR_F32);
    
    mma_direct_f32_fn[mi][ni](c, ldc, a, lda, b, ldb, Kc, clamp_min, clamp_max);
}

void tile_mma_direct(int Mc, int Nc, int Kc,
                   float* c, int ldc,
                   const float* a, int lda,
                   const float* b, int ldb,
                   float clamp_min, float clamp_max) {
    // TODO: tiled loop over Mc/Nc using mma_direct_f32_fn panels
}

// f16 — pack
void panel_mma_pack(int Mr, int Nc, int Kc,
                  half* c, int ldc,
                  const half* a, const half* b, int ldb,
                  float clamp_min, float clamp_max) {
    const int mi = panel_index(Mr, MR_F16);
    const int ni = panel_index(Nc, NR_F16);
    mma_pack_f16_fn[mi][ni](c, ldc, a, b, ldb, Kc, clamp_min, clamp_max);
}

void tile_mma_pack(int Mc, int Nc, int Kc,
                 half* c, int ldc,
                 const half* a, const half* b, int ldb,
                 float clamp_min, float clamp_max) {
    // TODO: tiled loop over Mc/Nc using mma_pack_f16_fn panels
}

// f16 — direct
void panel_mma_direct(int Mr, int Nc, int Kc,
                    half* c, int ldc,
                    const half* a, int lda,
                    const half* b, int ldb,
                    float clamp_min, float clamp_max) {
    const int mi = panel_index(Mr, MR_F16);
    const int ni = panel_index(Nc, NR_F16);
    mma_direct_f16_fn[mi][ni](c, ldc, a, lda, b, ldb, Kc, clamp_min, clamp_max);
}

void tile_mma_direct(int Mc, int Nc, int Kc,
                   half* c, int ldc,
                   const half* a, int lda,
                   const half* b, int ldb,
                   float clamp_min, float clamp_max) {
    // TODO: tiled loop over Mc/Nc using mma_direct_f16_fn panels
}

}  // namespace nnops::backend::cpu
