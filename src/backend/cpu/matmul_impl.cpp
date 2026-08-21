#include "matmul_impl.h"

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



// ---- pack dispatch tables (largest-first decomposition) -----------------

// f32 dispatch tables
// Row 0: MR-based (LHS packing), Row 1: NR-based (RHS packing)
constexpr std::array<std::array<PackF32Fn, 3>, 2> pack_trans_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f32, pack_trans_n4_f32, pack_trans_n1_f32},
    {pack_trans_n16_f32, pack_trans_n8_f32, pack_trans_n1_f32},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f32, pack_trans_n4_f32, pack_trans_n1_f32},
    {pack_trans_n12_f32, pack_trans_n4_f32, pack_trans_n1_f32},
#endif
}};

constexpr std::array<std::array<PackF32Fn, 3>, 2> pack_copy_f32_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_copy_n6_f32, pack_copy_n4_f32, pack_copy_n1_f32},
    {pack_copy_n16_f32, pack_copy_n8_f32, pack_copy_n1_f32},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_copy_n8_f32, pack_copy_n4_f32, pack_copy_n1_f32},
    {pack_copy_n12_f32, pack_copy_n4_f32, pack_copy_n1_f32},
#endif
}};

// f16 dispatch tables
// Row 0: MR-based (LHS packing), Row 1: NR-based (RHS packing)
constexpr std::array<std::array<PackF16Fn, 3>, 2> pack_trans_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_trans_n6_f16, pack_trans_n4_f16, pack_trans_n1_f16},
    {pack_trans_n16_f16, pack_trans_n8_f16, pack_trans_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_trans_n8_f16, pack_trans_n4_f16, pack_trans_n1_f16},
    {pack_trans_n16_f16, pack_trans_n8_f16, pack_trans_n1_f16},
#endif
}};

constexpr std::array<std::array<PackF16Fn, 3>, 2> pack_copy_f16_fn = {{
#ifdef NNOPS_ARCH_X86_64
    {pack_copy_n6_f16, pack_copy_n4_f16, pack_copy_n1_f16},
    {pack_copy_n16_f16, pack_copy_n8_f16, pack_copy_n1_f16},
#elif defined(NNOPS_ARCH_AARCH64)
    {pack_copy_n8_f16, pack_copy_n4_f16, pack_copy_n1_f16},
    {pack_copy_n16_f16, pack_copy_n8_f16, pack_copy_n1_f16},
#endif
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


// ---- tiled pack entry points -------------------------------------------
// ldd = aligned panel size in bytes: align_up(mr_max * kc * sizeof(T), 64)
// Panels are placed at panel_idx * ldd byte offsets, ensuring 64-byte-aligned starts.

void tile_pack_lhs(bool trans, int mc, int kc,
                   float* dst, int ldd,
                   const float* src, int lds,
                   float scale) {
    int pack_lds = trans ? 1 : lds;
    auto& pack_fns = trans ? pack_copy_f32_fn[0] : pack_trans_f32_fn[0];

    int panel_idx = 0;
    int m = 0;
    for(; m + MR_F32[0] <= mc; m += MR_F32[0], panel_idx++) {
        pack_fns[0](dst + panel_idx * ldd, src, lds, kc, scale);
        src += MR_F32[0] * pack_lds;
    }
    for(; m + MR_F32[1] <= mc; m += MR_F32[1], panel_idx++) {
        pack_fns[1](dst + panel_idx * ldd, src, lds, kc, scale);
        src += MR_F32[1] * pack_lds;
    }
    for(; m + MR_F32[2] <= mc; m += MR_F32[2], panel_idx++) {
        pack_fns[2](dst + panel_idx * ldd, src, lds, kc, scale);
        src += MR_F32[2] * pack_lds;
    }
}

void tile_pack_lhs(bool trans, int mc, int kc,
                  half* dst, int ldd,
                  const half* src, int lds, float scale) {

    int pack_lds = trans ? 1 : lds;
    auto& pack_fns = trans ? pack_copy_f16_fn[0] : pack_trans_f16_fn[0];

    int m = 0;
    for(; m + MR_F16[0] <= mc; m += MR_F16[0]) {
        pack_fns[0](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F16[0] * pack_lds;
    }
    for(; m + MR_F16[1] <= mc; m += MR_F16[1]) {
        pack_fns[1](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F16[1] * pack_lds;
    }
    for(; m + MR_F16[2] <= mc; m += MR_F16[2]) {
        pack_fns[2](dst, src, lds, kc, scale);
        dst += ldd;
        src += MR_F16[2] * pack_lds;
    }
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   float* dst, int ldd,
                   const float* src, int lds, float scale) {
    int pack_lds = trans ? lds : 1;

    auto& pack_fns = trans ? pack_trans_f32_fn[1] : pack_copy_f32_fn[1];

    int n = 0;
    for(; n + NR_F32[0] <= nc; n += NR_F32[0]) {
        pack_fns[0](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F32[0] * pack_lds;
    }
    for(; n + NR_F32[1] <= nc; n += NR_F32[1]) {
        pack_fns[1](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F32[1] * pack_lds;
    }
    for(; n + NR_F32[2] <= nc; n += NR_F32[2]) {
        pack_fns[2](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F32[2] * pack_lds;
    }
}

void tile_pack_rhs(bool trans, int nc, int kc,
                   half* dst, int ldd,
                   const half* src, int lds, float scale) {
    int pack_lds = trans ? lds : 1;

    auto& pack_fns = trans ? pack_trans_f16_fn[1] : pack_copy_f16_fn[1];

    int n = 0;
    for(; n + NR_F16[0] <= nc; n += NR_F16[0]) {
        pack_fns[0](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F16[0] * pack_lds;
    }
    for(; n + NR_F16[1] <= nc; n += NR_F16[1]) {
        pack_fns[1](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F16[1] * pack_lds;
    }
    for(; n + NR_F16[2] <= nc; n += NR_F16[2]) {
        pack_fns[2](dst, src, lds, kc, scale);
        dst += ldd;
        src += NR_F16[2] * pack_lds;
    }
}


// ---- mma entry points ---------------------------------------------------

// f32 — pack
// ldb = aligned panel stride in elements: align_up(NR_MAX * Kc * sizeof(float), 64) / sizeof(float)
void mrkcnc_mma_pack(int Nc, int Kc,
                     float* c, int ldc,
                     const float* packed_a, const float* b, int ldb,
                     float clamp_min, float clamp_max,
                     const std::array<MmaPackF32Fn, 3>& mma_pack_f32_fn) {
    int n = 0;
    for(; n + NR_F32[0] <= Nc; n += NR_F32[0]) {
        mma_pack_f32_fn[0](c + n, ldc, packed_a,
                           b, NR_F32[0], Kc, clamp_min, clamp_max);
        b += ldb;
    }

    for(; n + NR_F32[1] <= Nc; n += NR_F32[1]) {
        mma_pack_f32_fn[1](c + n, ldc, packed_a,
                           b, NR_F32[1], Kc, clamp_min, clamp_max);
        b += ldb;
    }

    for(; n + NR_F32[2] <= Nc; n += NR_F32[2]) {
        mma_pack_f32_fn[2](c + n, ldc, packed_a,
                           b, NR_F32[2], Kc, clamp_min, clamp_max);
        b += ldb;
    }
}

// f32 — direct
void mrkcnc_mma_direct(int Nc, int Kc,
                    float* c, int ldc,
                    const float* a, int lda,
                    const float* b, int ldb,
                    float clamp_min, float clamp_max,
                    const std::array<MmaDirectF32Fn, 3>& mma_direct_f32_fn) {
    int n = 0;
    for(; n + NR_F32[0] <= Nc; n += NR_F32[0]) {
        mma_direct_f32_fn[0](c + n, ldc, a, lda, b + n * ldb, ldb, Kc, clamp_min, clamp_max);
    }
    for(; n + NR_F32[1] <= Nc; n += NR_F32[1]) {
        mma_direct_f32_fn[1](c + n, ldc, a, lda, b + n * ldb, ldb, Kc, clamp_min, clamp_max);
    }
    for(; n + NR_F32[2] <= Nc; n += NR_F32[2]) {
        mma_direct_f32_fn[2](c + n, ldc, a, lda, b + n * ldb, ldb, Kc, clamp_min, clamp_max);
    }
}

// f16 — pack
// ldb = aligned panel stride in elements: align_up(NR_MAX * Kc * sizeof(half), 64) / sizeof(half)
void mrkcnc_mma_pack(int Nc, int Kc,
                     half* c, int ldc,
                     const half* packed_a, const half* b, int ldb,
                     float clamp_min, float clamp_max,
                     const std::array<MmaPackF16Fn, 3>& mma_pack_f16_fn) {
    int n = 0;
    for(; n + NR_F16[0] <= Nc; n += NR_F16[0]) {
        mma_pack_f16_fn[0](c + n, ldc, packed_a,
                           b, NR_F16[0], Kc, clamp_min, clamp_max);
        b += ldb;
    }

    for(; n + NR_F16[1] <= Nc; n += NR_F16[1]) {
        mma_pack_f16_fn[1](c + n, ldc, packed_a,
                           b, NR_F16[1], Kc, clamp_min, clamp_max);
        b += ldb;
    }

    for(; n + NR_F16[2] <= Nc; n += NR_F16[2]) {
        mma_pack_f16_fn[2](c + n, ldc, packed_a,
                           b, NR_F16[2], Kc, clamp_min, clamp_max);
        b += ldb;
    }
}

// f16 — direct
void mrkcnc_mma_direct(int Nc, int Kc,
                    half* c, int ldc,
                    const half* a, int lda,
                    const half* b, int ldb,
                    float clamp_min, float clamp_max,
                    const std::array<MmaDirectF16Fn, 3>& mma_direct_f16_fn) {
    int n = 0;
    for(; n + NR_F16[0] <= Nc; n += NR_F16[0]) {
        mma_direct_f16_fn[0](c + n, ldc, a, lda, b + n * ldb, ldb, Kc, clamp_min, clamp_max);
    }
    for(; n + NR_F16[1] <= Nc; n += NR_F16[1]) {
        mma_direct_f16_fn[1](c + n, ldc, a, lda, b + n * ldb, ldb, Kc, clamp_min, clamp_max);
    }
    for(; n + NR_F16[2] <= Nc; n += NR_F16[2]) {
        mma_direct_f16_fn[2](c + n, ldc, a, lda, b + n * ldb, ldb, Kc, clamp_min, clamp_max);
    }
}

void tile_mma_pack(int Mc, int Nc, int Kc,
                 float* c, int ldc,
                 const float* packed_a, int pack_lda, const float* b, int ldb,
                 float clamp_min, float clamp_max) {
    int panel_idx = 0;
    int m = 0;
    for(; m + MR_F32[0] <= Mc; m += MR_F32[0], panel_idx++) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a + panel_idx * pack_lda,
                        b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f32_fn[0]);
    }

    for(; m + MR_F32[1] <= Mc; m += MR_F32[1], panel_idx++) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a + panel_idx * pack_lda,
                        b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f32_fn[1]);
    }

    for(; m + MR_F32[2] <= Mc; m += MR_F32[2], panel_idx++) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a + panel_idx * pack_lda,
                        b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f32_fn[2]);
    }
}


void tile_mma_direct(int Mc, int Nc, int Kc,
                   float* c, int ldc,
                   const float* a, int lda,
                   const float* b, int ldb,
                   float clamp_min, float clamp_max) {
    int m = 0;
    for(; m + MR_F32[0] <= Mc; m += MR_F32[0]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f32_fn[0]);
    }
    for(; m + MR_F32[1] <= Mc; m += MR_F32[1]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f32_fn[1]);
    }
    for(; m + MR_F32[2] <= Mc; m += MR_F32[2]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f32_fn[2]);
    }
}

void tile_mma_pack(int Mc, int Nc, int Kc,
                 half* c, int ldc,
                 const half* packed_a, int pack_lda, const half* b, int ldb,
                 float clamp_min, float clamp_max) {
    int panel_idx = 0;
    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0], panel_idx++) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a + panel_idx * pack_lda,
                        b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f16_fn[0]);
    }

    for(; m + MR_F16[1] <= Mc; m += MR_F16[1], panel_idx++) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a + panel_idx * pack_lda,
                        b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f16_fn[1]);
    }

    for(; m + MR_F16[2] <= Mc; m += MR_F16[2], panel_idx++) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a + panel_idx * pack_lda,
                        b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f16_fn[2]);
    }
}


void tile_mma_direct(int Mc, int Nc, int Kc,
                   half* c, int ldc,
                   const half* a, int lda,
                   const half* b, int ldb,
                   float clamp_min, float clamp_max) {
    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f16_fn[0]);
    }
    for(; m + MR_F16[1] <= Mc; m += MR_F16[1]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f16_fn[1]);
    }
    for(; m + MR_F16[2] <= Mc; m += MR_F16[2]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f16_fn[2]);
    }
}


}  // namespace nnops::backend::cpu
