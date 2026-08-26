#include "matmul_helper.h"

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
// ldb < 0: B packed (mma_ldb=NR[i], b_offset=64B-aligned panel stride)
// else:    B raw    (mma_ldb=ldb,    b_offset=ldb)
void mrkcnc_mma_pack(int Nc, int Kc,
                     float* c, int ldc,
                     const float* packed_a, 
                     const float* maybe_packed_b, int ldb,
                     float clamp_min, float clamp_max,
                     const std::array<MmaPackF32Fn, 3>& mma_pack_f32_fn) {
    bool is_b_packed = ldb < 0;
    int n = 0;
    for(; n + NR_F32[0] <= Nc; n += NR_F32[0]) {
        int mma_ldb = is_b_packed ? NR_F32[0] : ldb;
        mma_pack_f32_fn[0](c + n, ldc, packed_a,
                           maybe_packed_b, mma_ldb, 
                           Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F32[0] * Kc * sizeof(float)) / sizeof(float) : ldb;
        maybe_packed_b += b_offset;
    }

    for(; n + NR_F32[1] <= Nc; n += NR_F32[1]) {
        int mma_ldb = is_b_packed ? NR_F32[1] : ldb;
        mma_pack_f32_fn[1](c + n, ldc, packed_a,
                           maybe_packed_b, mma_ldb, 
                           Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F32[1] * Kc * sizeof(float)) / sizeof(float) : ldb;
        maybe_packed_b += b_offset;
    }

    for(; n + NR_F32[2] <= Nc; n += NR_F32[2]) {
        int mma_ldb = is_b_packed ? NR_F32[2] : ldb;
        mma_pack_f32_fn[2](c + n, ldc, packed_a,
                           maybe_packed_b, mma_ldb,
                           Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F32[2] * Kc * sizeof(float)) / sizeof(float) : ldb;
        maybe_packed_b += b_offset;
    }
}

// f32 — direct
// ldb < 0: B packed (mma_ldb=NR[i], b_offset=64B-aligned panel stride)
// else:    B raw    (mma_ldb=ldb,    b_offset=ldb)
void mrkcnc_mma_direct(int Nc, int Kc,
                    float* c, int ldc,
                    const float* a, int lda,
                    const float* maybe_packed_b, int ldb,
                    float clamp_min, float clamp_max,
                    const std::array<MmaDirectF32Fn, 3>& mma_direct_f32_fn) {
    bool is_b_packed = ldb < 0;
    
    int n = 0;
    for(; n + NR_F32[0] <= Nc; n += NR_F32[0]) {
        int mma_ldb = is_b_packed ? NR_F32[0] : ldb;
        mma_direct_f32_fn[0](c + n, ldc, a, lda, maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F32[0] * Kc * sizeof(float)) / sizeof(float) : ldb;
        maybe_packed_b += b_offset;
    }
    for(; n + NR_F32[1] <= Nc; n += NR_F32[1]) {
        int mma_ldb = is_b_packed ? NR_F32[1] : ldb;
        mma_direct_f32_fn[1](c + n, ldc, a, lda, maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F32[1] * Kc * sizeof(float)) / sizeof(float) : ldb;
        maybe_packed_b += b_offset;
    }
    for(; n + NR_F32[2] <= Nc; n += NR_F32[2]) {
        int mma_ldb = is_b_packed ? NR_F32[2] : ldb;
        mma_direct_f32_fn[2](c + n, ldc, a, lda, maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F32[2] * Kc * sizeof(float)) / sizeof(float) : ldb;
        maybe_packed_b += b_offset;
    }
}

// f16 — pack
// ldb < 0: B packed (mma_ldb=NR[i], b_offset=64B-aligned panel stride)
// else:    B raw    (mma_ldb=ldb,    b_offset=ldb)
void mrkcnc_mma_pack(int Nc, int Kc,
                     half* c, int ldc,
                     const half* packed_a, const half* maybe_packed_b, int ldb,
                     float clamp_min, float clamp_max,
                     const std::array<MmaPackF16Fn, 3>& mma_pack_f16_fn) {
    bool is_b_packed = ldb < 0;
    int n = 0;
    for(; n + NR_F16[0] <= Nc; n += NR_F16[0]) {
        int mma_ldb = is_b_packed ? NR_F16[0] : ldb;
        mma_pack_f16_fn[0](c + n, ldc, packed_a,
                           maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F16[0] * Kc * sizeof(half)) / sizeof(half) : ldb;
        maybe_packed_b += b_offset;
    }

    for(; n + NR_F16[1] <= Nc; n += NR_F16[1]) {
        int mma_ldb = is_b_packed ? NR_F16[1] : ldb;
        mma_pack_f16_fn[1](c + n, ldc, packed_a,
                           maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F16[1] * Kc * sizeof(half)) / sizeof(half) : ldb;
        maybe_packed_b += b_offset;
    }

    for(; n + NR_F16[2] <= Nc; n += NR_F16[2]) {
        int mma_ldb = is_b_packed ? NR_F16[2] : ldb;
        mma_pack_f16_fn[2](c + n, ldc, packed_a,
                           maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F16[2] * Kc * sizeof(half)) / sizeof(half) : ldb;
        maybe_packed_b += b_offset;
    }
}

// f16 — direct
// ldb < 0: B packed (mma_ldb=NR[i], b_offset=64B-aligned panel stride)
// else:    B raw    (mma_ldb=ldb,    b_offset=ldb)
void mrkcnc_mma_direct(int Nc, int Kc,
                    half* c, int ldc,
                    const half* a, int lda,
                    const half* maybe_packed_b, int ldb,
                    float clamp_min, float clamp_max,
                    const std::array<MmaDirectF16Fn, 3>& mma_direct_f16_fn) {
    bool is_b_packed = ldb < 0;

    int n = 0;
    for(; n + NR_F16[0] <= Nc; n += NR_F16[0]) {
        int mma_ldb = is_b_packed ? NR_F16[0] : ldb;
        mma_direct_f16_fn[0](c + n, ldc, a, lda, maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F16[0] * Kc * sizeof(half)) / sizeof(half) : ldb;
        maybe_packed_b += b_offset;
    }
    for(; n + NR_F16[1] <= Nc; n += NR_F16[1]) {
        int mma_ldb = is_b_packed ? NR_F16[1] : ldb;
        mma_direct_f16_fn[1](c + n, ldc, a, lda, maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F16[1] * Kc * sizeof(half)) / sizeof(half) : ldb;
        maybe_packed_b += b_offset;
    }
    for(; n + NR_F16[2] <= Nc; n += NR_F16[2]) {
        int mma_ldb = is_b_packed ? NR_F16[2] : ldb;
        mma_direct_f16_fn[2](c + n, ldc, a, lda, maybe_packed_b, mma_ldb, Kc, clamp_min, clamp_max);
        int b_offset = is_b_packed ? align_up<PANEL_ALIGN_BYTES>(NR_F16[2] * Kc * sizeof(half)) / sizeof(half) : ldb;
        maybe_packed_b += b_offset;
    }
}

// f32 — pack
// packed_a stride = align_up(MR[i] * Kc * sizeof(float), PANEL_ALIGN_BYTES) / sizeof(float)
void tile_mma_pack(int Mc, int Nc, int Kc,
                 float* c, int ldc,
                 const float* packed_a, const float* maybe_packed_b, int ldb,
                 float clamp_min, float clamp_max) {
   
    int m = 0;
    for(; m + MR_F32[0] <= Mc; m += MR_F32[0]) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f32_fn[0]);
        packed_a += align_up<PANEL_ALIGN_BYTES>(MR_F32[0] * Kc * sizeof(float)) / sizeof(float);
    }

    for(; m + MR_F32[1] <= Mc; m += MR_F32[1]) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f32_fn[1]);
        packed_a += align_up<PANEL_ALIGN_BYTES>(MR_F32[1] * Kc * sizeof(float)) / sizeof(float);
    }

    for(; m + MR_F32[2] <= Mc; m += MR_F32[2]) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f32_fn[2]);
        packed_a += align_up<PANEL_ALIGN_BYTES>(MR_F32[2] * Kc * sizeof(float)) / sizeof(float);
    }
}


// f32 — direct
// A is raw (indexed via lda), B may be packed (ldb < 0) or raw
void tile_mma_direct(int Mc, int Nc, int Kc,
                   float* c, int ldc,
                   const float* a, int lda,
                   const float* maybe_packed_b, int ldb,
                   float clamp_min, float clamp_max) {
    int m = 0;
    for(; m + MR_F32[0] <= Mc; m += MR_F32[0]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          maybe_packed_b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f32_fn[0]);
    }
    for(; m + MR_F32[1] <= Mc; m += MR_F32[1]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          maybe_packed_b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f32_fn[1]);
    }
    for(; m + MR_F32[2] <= Mc; m += MR_F32[2]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          maybe_packed_b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f32_fn[2]);
    }
}

// f16 — pack
// packed_a stride = align_up(MR[i] * Kc * sizeof(half), PANEL_ALIGN_BYTES) / sizeof(half)
void tile_mma_pack(int Mc, int Nc, int Kc,
                 half* c, int ldc,
                 const half* packed_a, const half* maybe_packed_b, int ldb,
                 float clamp_min, float clamp_max) {

    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0]) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f16_fn[0]);
        packed_a += align_up<PANEL_ALIGN_BYTES>(MR_F16[0] * Kc * sizeof(half)) / sizeof(half);
    }

    for(; m + MR_F16[1] <= Mc; m += MR_F16[1]) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f16_fn[1]);
        packed_a += align_up<PANEL_ALIGN_BYTES>(MR_F16[1] * Kc * sizeof(half)) / sizeof(half);
    }

    for(; m + MR_F16[2] <= Mc; m += MR_F16[2]) {
        mrkcnc_mma_pack(Nc, Kc,
                        c + m * ldc, ldc,
                        packed_a, maybe_packed_b, ldb,
                        clamp_min, clamp_max,
                        mma_pack_f16_fn[2]);
        packed_a += align_up<PANEL_ALIGN_BYTES>(MR_F16[2] * Kc * sizeof(half)) / sizeof(half);
    }
}


// f16 — direct
// A is raw (indexed via lda), B may be packed (ldb < 0) or raw
void tile_mma_direct(int Mc, int Nc, int Kc,
                   half* c, int ldc,
                   const half* a, int lda,
                   const half* maybe_packed_b, int ldb,
                   float clamp_min, float clamp_max) {
    int m = 0;
    for(; m + MR_F16[0] <= Mc; m += MR_F16[0]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          maybe_packed_b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f16_fn[0]);
    }
    for(; m + MR_F16[1] <= Mc; m += MR_F16[1]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          maybe_packed_b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f16_fn[1]);
    }
    for(; m + MR_F16[2] <= Mc; m += MR_F16[2]) {
        mrkcnc_mma_direct(Nc, Kc,
                          c + m * ldc, ldc,
                          a + m * lda, lda,
                          maybe_packed_b, ldb,
                          clamp_min, clamp_max,
                          mma_direct_f16_fn[2]);
    }
}


#include "nnops/detail/half.hpp"
#include "nnops/detail/simd.hpp"

// ---- tile scale -----------------------------------------------------------
// SIMD-accelerated in-place scale: C[i] *= scale
// Uses 8-wide SIMD lanes (v_f32x8 / v_f16x8) with scalar tail for remainder.

template <typename T>
void tile_scale_impl(T* c, int ldc, float scale, int M, int N) {
    using namespace nnops::simd;
    constexpr int L = simd_lane_for<T>;  // 8 for both f32 and f16
    for (int m = 0; m < M; ++m) {
        T* row = c + m * ldc;
        auto v_scale = v_set1(row, scale);
        int i = 0;
        for (; i + L <= N; i += L) {
            v_store(row + i, v_mul(v_load(row + i), v_scale));
        }
        for (; i < N; ++i) {
            s_store(&row[i], s_load(&row[i]) * scale);
        }
    }
}

void tile_scale(float* c, int ldc, float scale, int M, int N) {
    tile_scale_impl(c, ldc, scale, M, N);
}

void tile_scale(half* c, int ldc, float scale, int M, int N) {
    tile_scale_impl(c, ldc, scale, M, N);
}



}  // namespace nnops::backend::cpu
