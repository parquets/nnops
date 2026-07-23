#pragma once
/// @file imatmul.h
/// @brief Platform-abstracted internal MatMul interface.
///
/// Three-layer architecture (all in one file, separated by section):
///   1. TILED PACK   — mr/nr constants, pack tables, pack_lhs_panel / pack_rhs_panel
///   2. TILED MMA    — MMA tables, mma_tile
///   3. GEMM         — glue: gemm_tiled_f32  (pack + mma + epilogue)
///
/// Shields callers from ARM vs x86_64 MR/NR differences through uniform
/// function-pointer tables. The tiling logic is identical across platforms —
/// only the tables differ.
///
/// Reference pattern: nn_compute TiledMatMulInterface +
///   FregmentMr / FregmentNr (onnxruntime_plugin compute/cpu/matmul.hpp)

#include <cstdint>
#include <cfloat>
#include <cstring>
#include <vector>
#include <type_traits>

#include "nnops/core/epilogue.hpp"
#include "nnops/core/compute_context.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/ops/matmul.hpp"
#include "backend/cpu/common/restrict.hpp"

// ---- Arch-specific micro-kernel includes ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)
#include "backend/cpu/x86_64/pack_f32.hpp"
#include "backend/cpu/x86_64/mma_pack_f32.hpp"
#include "backend/cpu/x86_64/pack_f16.hpp"
#include "backend/cpu/x86_64/mma_pack_f16.hpp"
#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)
#include "backend/cpu/aarch64/pack_f32.hpp"
#include "backend/cpu/aarch64/mma_pack_f32.hpp"
#include "backend/cpu/aarch64/pack_f16.hpp"
#include "backend/cpu/aarch64/mma_pack_f16.hpp"
#endif

#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu {

// ---- FP16 type bridge (arch-dependent) ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)
using half_t = half;
#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)
using half_t = float16_t;
#else
using half_t = float;  // fallback (unused)
#endif

// =========================================================================
// 1. TILED PACK  -  mr/nr constants, pack tables, pack panel dispatch
// =========================================================================

// ---- 1a. Platform-specific micro-panel sizes ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)
inline constexpr int mr_f32[3] = {6, 4, 1};
inline constexpr int nr_f32[3] = {16, 8, 1};
inline constexpr int mr_max_f32 = 6;
inline constexpr int nr_max_f32 = 16;
#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)
inline constexpr int mr_f32[3] = {8, 4, 1};
inline constexpr int nr_f32[3] = {12, 4, 1};
inline constexpr int mr_max_f32 = 8;
inline constexpr int nr_max_f32 = 12;
#else
inline constexpr int mr_f32[3] = {1, 1, 1};
inline constexpr int nr_f32[3] = {1, 1, 1};
inline constexpr int mr_max_f32 = 1;
inline constexpr int nr_max_f32 = 1;
#endif

// ---- 1b. Pack function pointer types ----

/// LHS pack (transpose): reads N rows, transposes N×K_step blocks, scales.
using PackTransFn = void (*)(float* NNOPS_RESTRICT output,
                              const float* NNOPS_RESTRICT input,
                              int ir_step, int K, float scale) noexcept;

/// RHS pack (copy): reads K rows of N elements, scales, writes contiguously.
using PackCopyFn = void (*)(float* NNOPS_RESTRICT output,
                             const float* NNOPS_RESTRICT input,
                             int ir_step, int K, float scale) noexcept;

// ---- 1c. Pack tables (arch-specific, compile-time dispatch) ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)

inline constexpr PackTransFn pack_lhs_trans_tbl[3] = {
    x86_64::pack_trans_n6_f32,
    x86_64::pack_trans_n4_f32,
    x86_64::pack_trans_n1_f32,
};

inline constexpr PackCopyFn pack_lhs_copy_tbl[3] = {
    x86_64::pack_copy_n6_f32,
    x86_64::pack_copy_n4_f32,
    x86_64::pack_copy_n1_f32,
};

inline constexpr PackCopyFn pack_rhs_copy_tbl[3] = {
    x86_64::pack_copy_n16_f32,
    x86_64::pack_copy_n8_f32,
    x86_64::pack_copy_n1_f32,
};

inline constexpr PackTransFn pack_rhs_trans_tbl[3] = {
    x86_64::pack_trans_n16_f32,
    x86_64::pack_trans_n8_f32,
    x86_64::pack_trans_n1_f32,
};

#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)

inline constexpr PackTransFn pack_lhs_trans_tbl[3] = {
    aarch64::pack_trans_n8_f32,
    aarch64::pack_trans_n4_f32,
    aarch64::pack_trans_n1_f32,
};

inline constexpr PackCopyFn pack_lhs_copy_tbl[3] = {
    aarch64::pack_copy_n8_f32,
    aarch64::pack_copy_n4_f32,
    aarch64::pack_copy_n1_f32,
};

inline constexpr PackCopyFn pack_rhs_copy_tbl[3] = {
    aarch64::pack_copy_n12_f32,
    aarch64::pack_copy_n4_f32,
    aarch64::pack_copy_n1_f32,
};

inline constexpr PackTransFn pack_rhs_trans_tbl[3] = {
    aarch64::pack_trans_n12_f32,
    aarch64::pack_trans_n4_f32,
    aarch64::pack_trans_n1_f32,
};

#else
inline constexpr PackTransFn pack_lhs_trans_tbl[3]  = {};
inline constexpr PackCopyFn  pack_lhs_copy_tbl[3]   = {};
inline constexpr PackCopyFn  pack_rhs_copy_tbl[3]   = {};
inline constexpr PackTransFn pack_rhs_trans_tbl[3]  = {};
#endif

// ---- 1d. Tier index helpers ----

inline int m_idx_for(int mr) noexcept {
    return (mr == mr_f32[0]) ? 0 : (mr == mr_f32[1]) ? 1 : 2;
}

inline int n_idx_for(int nr) noexcept {
    return (nr == nr_f32[0]) ? 0 : (nr == nr_f32[1]) ? 1 : 2;
}

inline int choose_mr(int64_t rem) noexcept {
    if (rem >= mr_f32[0]) return mr_f32[0];
    if (rem >= mr_f32[1]) return mr_f32[1];
    return mr_f32[2];
}

inline int choose_nr(int64_t rem) noexcept {
    if (rem >= nr_f32[0]) return nr_f32[0];
    if (rem >= nr_f32[1]) return nr_f32[1];
    return nr_f32[2];
}

// ---- 1e. Pack panel entry points ----

/// Pack one mr-panel of A into pack_A.
///
/// When trans=true:  reads mr rows from a_panel, transposes to K×mr layout
///                   (standard MMA format). Uses pack_lhs_trans_tbl.
/// When trans=false: copies mr rows of K elements contiguously without
///                   transposing. Uses pack_lhs_copy_tbl.
///
/// @param mr      Panel height (must be one of mr_f32[0..2]).
/// @param K       Inner dimension.
/// @param pack_A  Output buffer, size mr * K.
/// @param a_panel Pointer into A — start of row m (or column m if transposed).
/// @param ir_step Stride between consecutive input rows (= lda).
/// @param trans   If true, transpose during pack; otherwise copy.
inline void tiled_pack_lhs(int mr, int K,
                           float* NNOPS_RESTRICT pack_A,
                           const float* NNOPS_RESTRICT a_panel,
                           int ir_step, bool trans) noexcept {
    const int m_i = m_idx_for(mr);
    if (trans) {
        pack_lhs_trans_tbl[m_i](pack_A, a_panel, ir_step, K, 1.0f);
    } else {
        pack_lhs_copy_tbl[m_i](pack_A, a_panel, ir_step, K, 1.0f);
    }
}

/// Pack one nr-panel of B into pack_B.
///
/// When trans=true:  B is N×K layout; reads nr rows and transposes to K×nr
///                   (standard MMA format). Uses pack_rhs_trans_tbl.
/// When trans=false: B is K×N layout; copies nr-element columns from K rows
///                   contiguously. Uses pack_rhs_copy_tbl.
///
/// @param nr       Panel width (must be one of nr_f32[0..2]).
/// @param K        Inner dimension.
/// @param pack_B   Output buffer, size nr * K.
/// @param b_panel  Pointer into B — start of column n (or row n if transposed).
/// @param ir_step  Stride between consecutive input rows (= ldb).
/// @param trans    If true, B is N×K and we pack B[n:n+nr, :] via transpose.
inline void tiled_pack_rhs(int nr, int K,
                           float* NNOPS_RESTRICT pack_B,
                           const float* NNOPS_RESTRICT b_panel,
                           int ir_step, bool trans) noexcept {
    const int n_i = n_idx_for(nr);
    if (trans) {
        pack_rhs_trans_tbl[n_i](pack_B, b_panel, ir_step, K, 1.0f);
    } else {
        pack_rhs_copy_tbl[n_i](pack_B, b_panel, ir_step, K, 1.0f);
    }
}

// ---- 1f. FP16 pack function pointer types ----

/// LHS pack for fp16 (transpose).
using PackTransFnF16 = void (*)(half_t* NNOPS_RESTRICT output,
                                 const half_t* NNOPS_RESTRICT input,
                                 int ir_step, int K, float scale) noexcept;

/// RHS pack for fp16 (copy).
using PackCopyFnF16 = void (*)(half_t* NNOPS_RESTRICT output,
                                const half_t* NNOPS_RESTRICT input,
                                int ir_step, int K, float scale) noexcept;

// ---- 1g. FP16 platform-specific micro-panel sizes ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)
inline constexpr int mr_f16[3] = {6, 4, 1};
inline constexpr int nr_f16[3] = {16, 8, 1};
inline constexpr int mr_max_f16 = 6;
inline constexpr int nr_max_f16 = 16;
#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)
inline constexpr int mr_f16[3] = {8, 4, 1};
inline constexpr int nr_f16[3] = {16, 8, 1};
inline constexpr int mr_max_f16 = 8;
inline constexpr int nr_max_f16 = 16;
#else
inline constexpr int mr_f16[3] = {1, 1, 1};
inline constexpr int nr_f16[3] = {1, 1, 1};
inline constexpr int mr_max_f16 = 1;
inline constexpr int nr_max_f16 = 1;
#endif

// ---- 1h. FP16 pack tables (arch-specific, compile-time dispatch) ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)

inline constexpr PackTransFnF16 pack_lhs_trans_tbl_f16[3] = {
    x86_64::pack_trans_n6_f16,
    x86_64::pack_trans_n4_f16,
    x86_64::pack_trans_n1_f16,
};

inline constexpr PackCopyFnF16 pack_lhs_copy_tbl_f16[3] = {
    x86_64::pack_copy_n6_f16,
    x86_64::pack_copy_n4_f16,
    x86_64::pack_copy_n1_f16,
};

inline constexpr PackCopyFnF16 pack_rhs_copy_tbl_f16[3] = {
    x86_64::pack_copy_n16_f16,
    x86_64::pack_copy_n8_f16,
    x86_64::pack_copy_n1_f16,
};

inline constexpr PackTransFnF16 pack_rhs_trans_tbl_f16[3] = {
    x86_64::pack_trans_n16_f16,
    x86_64::pack_trans_n8_f16,
    x86_64::pack_trans_n1_f16,
};

#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)

inline constexpr PackTransFnF16 pack_lhs_trans_tbl_f16[3] = {
    aarch64::pack_trans_n8_f16,
    aarch64::pack_trans_n4_f16,
    aarch64::pack_trans_n1_f16,
};

inline constexpr PackCopyFnF16 pack_lhs_copy_tbl_f16[3] = {
    aarch64::pack_copy_n8_f16,
    aarch64::pack_copy_n4_f16,
    aarch64::pack_copy_n1_f16,
};

inline constexpr PackCopyFnF16 pack_rhs_copy_tbl_f16[3] = {
    aarch64::pack_copy_n16_f16,
    aarch64::pack_copy_n8_f16,
    aarch64::pack_copy_n1_f16,
};

inline constexpr PackTransFnF16 pack_rhs_trans_tbl_f16[3] = {
    aarch64::pack_trans_n16_f16,
    aarch64::pack_trans_n8_f16,
    aarch64::pack_trans_n1_f16,
};

#else
inline constexpr PackTransFnF16 pack_lhs_trans_tbl_f16[3]  = {};
inline constexpr PackCopyFnF16  pack_lhs_copy_tbl_f16[3]   = {};
inline constexpr PackCopyFnF16  pack_rhs_copy_tbl_f16[3]   = {};
inline constexpr PackTransFnF16 pack_rhs_trans_tbl_f16[3]  = {};
#endif

// ---- 1i. FP16 tier index helpers ----

inline int m_idx_for_f16(int mr) noexcept {
    return (mr == mr_f16[0]) ? 0 : (mr == mr_f16[1]) ? 1 : 2;
}

inline int n_idx_for_f16(int nr) noexcept {
    return (nr == nr_f16[0]) ? 0 : (nr == nr_f16[1]) ? 1 : 2;
}

inline int choose_mr_f16(int64_t rem) noexcept {
    if (rem >= mr_f16[0]) return mr_f16[0];
    if (rem >= mr_f16[1]) return mr_f16[1];
    return mr_f16[2];
}

inline int choose_nr_f16(int64_t rem) noexcept {
    if (rem >= nr_f16[0]) return nr_f16[0];
    if (rem >= nr_f16[1]) return nr_f16[1];
    return nr_f16[2];
}

// ---- 1j. FP16 pack panel entry points ----

/// Pack one mr-panel of A (fp16) into pack_A.
///
/// When trans=true:  transposes to K×mr layout (standard MMA format).
/// When trans=false: copies mr rows of K elements contiguously.
inline void tiled_pack_lhs_f16(int mr, int K,
                                half_t* NNOPS_RESTRICT pack_A,
                                const half_t* NNOPS_RESTRICT a_panel,
                                int ir_step, bool trans) noexcept {
    const int m_i = m_idx_for_f16(mr);
    if (trans) {
        pack_lhs_trans_tbl_f16[m_i](pack_A, a_panel, ir_step, K, 1.0f);
    } else {
        pack_lhs_copy_tbl_f16[m_i](pack_A, a_panel, ir_step, K, 1.0f);
    }
}

/// Pack one nr-panel of B (fp16) into pack_B.
///
/// When trans=true:  B is N×K; reads nr rows and transposes to K×nr.
/// When trans=false: B is K×N; copies nr-element columns contiguously.
inline void tiled_pack_rhs_f16(int nr, int K,
                                half_t* NNOPS_RESTRICT pack_B,
                                const half_t* NNOPS_RESTRICT b_panel,
                                int ir_step, bool trans) noexcept {
    const int n_i = n_idx_for_f16(nr);
    if (trans) {
        pack_rhs_trans_tbl_f16[n_i](pack_B, b_panel, ir_step, K, 1.0f);
    } else {
        pack_rhs_copy_tbl_f16[n_i](pack_B, b_panel, ir_step, K, 1.0f);
    }
}

// =========================================================================
// 2. TILED MMA  -  MMA tables, single-tile dispatch
// =========================================================================

// ---- 2a. MMA function pointer type ----

/// MMA micro-kernel (packed A):
///   C[mr][nr] += A_packed[mr][K] × B_packed[nr][K]  then clamp.
using MmaPackFn = void (*)(float* NNOPS_RESTRICT C, int ldc,
                            const float* NNOPS_RESTRICT A,
                            const float* NNOPS_RESTRICT B,
                            int ldb, int K,
                            float clamp_min, float clamp_max) noexcept;

// ---- 2b. MMA tables (arch-specific) ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)

// x86_64: mr ∈ {6,4,1} × nr ∈ {16,8,1}
inline constexpr MmaPackFn mma_pack_tbl[3][3] = {
    {x86_64::mma_pack_6x16_f32, x86_64::mma_pack_6x8_f32, x86_64::mma_pack_6x1_f32},
    {x86_64::mma_pack_4x16_f32, x86_64::mma_pack_4x8_f32, x86_64::mma_pack_4x1_f32},
    {x86_64::mma_pack_1x16_f32, x86_64::mma_pack_1x8_f32, x86_64::mma_pack_1x1_f32},
};

#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)

// AArch64: mr ∈ {8,4,1} × nr ∈ {12,4,1}
inline constexpr MmaPackFn mma_pack_tbl[3][3] = {
    {aarch64::mma_pack_8x12_f32, aarch64::mma_pack_8x4_f32, aarch64::mma_pack_8x1_f32},
    {aarch64::mma_pack_4x12_f32, aarch64::mma_pack_4x4_f32, aarch64::mma_pack_4x1_f32},
    {aarch64::mma_pack_1x12_f32, aarch64::mma_pack_1x4_f32, aarch64::mma_pack_1x1_f32},
};

#else
inline constexpr MmaPackFn mma_pack_tbl[3][3] = {};
#endif

// ---- 2c. MMA tile entry point ----

/// Compute one mr×nr tile:  C += A_packed × B_packed.
///
/// @param mr, nr    Tile dimensions (must map to valid table entries).
/// @param K         Inner dimension.
/// @param C         Output sub-matrix, mr × ldc (row-major).
/// @param ldc       Row stride of C.
/// @param pack_A    Packed A buffer, mr × K.
/// @param pack_B    Packed B buffer, nr × K.
/// @param ldb       Column stride of B in packed buffer (= nr for contiguous).
/// @param clamp_min,max  Value clamp range after accumulation.
inline void mma_tile(int mr, int nr, int K,
                     float* NNOPS_RESTRICT C, int ldc,
                     const float* NNOPS_RESTRICT pack_A,
                     const float* NNOPS_RESTRICT pack_B,
                     int ldb,
                     float clamp_min, float clamp_max) noexcept {
    mma_pack_tbl[m_idx_for(mr)][n_idx_for(nr)](
        C, ldc, pack_A, pack_B, ldb, K, clamp_min, clamp_max);
}

// ---- 2d. FP16 MMA function pointer type ----

/// MMA micro-kernel for fp16 (packed A):
///   C[mr][nr] += A_packed[mr][K] × B_packed[nr][K]  then clamp.
using MmaPackFnF16 = void (*)(half_t* NNOPS_RESTRICT C, int ldc,
                               const half_t* NNOPS_RESTRICT A,
                               const half_t* NNOPS_RESTRICT B,
                               int ldb, int K,
                               float clamp_min, float clamp_max) noexcept;

// ---- 2e. FP16 MMA tables (arch-specific) ----
#if defined(__x86_64__) || defined(_M_X64) || defined(NNOPS_ARCH_X86_64)

// x86_64 f16: mr ∈ {6,4,1} × nr ∈ {16,8,1}
inline constexpr MmaPackFnF16 mma_pack_tbl_f16[3][3] = {
    {x86_64::mma_pack_6x16_f16, x86_64::mma_pack_6x8_f16, x86_64::mma_pack_6x1_f16},
    {x86_64::mma_pack_4x16_f16, x86_64::mma_pack_4x8_f16, x86_64::mma_pack_4x1_f16},
    {x86_64::mma_pack_1x16_f16, x86_64::mma_pack_1x8_f16, x86_64::mma_pack_1x1_f16},
};

#elif defined(__aarch64__) || defined(NNOPS_ARCH_AARCH64)

// AArch64 f16: mr ∈ {8,4,1} × nr ∈ {16,8,1}
inline constexpr MmaPackFnF16 mma_pack_tbl_f16[3][3] = {
    {aarch64::mma_pack_8x16_f16, aarch64::mma_pack_8x8_f16, aarch64::mma_pack_8x1_f16},
    {aarch64::mma_pack_4x16_f16, aarch64::mma_pack_4x8_f16, aarch64::mma_pack_4x1_f16},
    {aarch64::mma_pack_1x16_f16, aarch64::mma_pack_1x8_f16, aarch64::mma_pack_1x1_f16},
};

#else
inline constexpr MmaPackFnF16 mma_pack_tbl_f16[3][3] = {};
#endif

// ---- 2f. FP16 MMA tile entry point ----

/// Compute one mr×nr tile for fp16:  C += A_packed × B_packed.
inline void mma_tile_f16(int mr, int nr, int K,
                          half_t* NNOPS_RESTRICT C, int ldc,
                          const half_t* NNOPS_RESTRICT pack_A,
                          const half_t* NNOPS_RESTRICT pack_B,
                          int ldb,
                          float clamp_min, float clamp_max) noexcept {
    mma_pack_tbl_f16[m_idx_for_f16(mr)][n_idx_for_f16(nr)](
        C, ldc, pack_A, pack_B, ldb, K, clamp_min, clamp_max);
}

// =========================================================================
// 3. GEMM  -  glue: epilogue + 2D tiled GEMM + batch kernel
// =========================================================================

// ---- 3a. Epilogue block apply ----

/// Apply epilogue (non-clamp types) to an mr×nr output block.
inline void apply_epilogue_block(float* C, int64_t ldc, int mr, int nr,
                                  const Epilogue& ep) noexcept {
    for (int m = 0; m < mr; ++m) {
        for (int n = 0; n < nr; ++n) {
            C[m * ldc + n] = apply_epilogue(ep, C[m * ldc + n], n);
        }
    }
}

// ---- 3b. Tiled 2D GEMM ----

/// Tiled float32 GEMM:  C[M][N] = A[M][K] × B[K][N].
///
/// Decomposes M→mr panels, N→nr panels. For each (mr,nr) tile:
///   1. pack_lhs_panel()  — pack A[m:m+mr, :]
///   2. pack_rhs_panel()  — pack B[:, n:n+nr]
///   3. mma_tile()        — accumulate C += packed_A × packed_B
///   4. epilogue          — apply activation (if not fused into clamp)
///
/// @param C            Output, M × N row-major, ldc stride.
/// @param ldc          Row stride of C.
/// @param A            Input A.
/// @param lda          Row stride of A.
/// @param B            Input B.
/// @param ldb          Row stride of B.
/// @param M, N, K      Matrix dimensions.
/// @param transpose_a  If true, A is stored as K×M.
/// @param transpose_b  If true, B is stored as N×K.
/// @param epilogue     Post-processing activation.
/// @param add_to       If true, accumulate into existing C.
/// @param ctx          Compute context (reserved for future parallelism).
inline void gemm_tiled_f32(
    float* C, int64_t ldc,
    const float* A, int64_t lda,
    const float* B, int64_t ldb,
    int64_t M, int64_t N, int64_t K,
    bool transpose_a, bool transpose_b,
    const Epilogue& epilogue,
    bool add_to,
    const ComputeContext& /*ctx*/)
{
    // ---- zero output if overwriting ----
    if (!add_to && M > 0 && N > 0) {
        if (ldc == N) {
            std::memset(C, 0, static_cast<size_t>(M * N) * sizeof(float));
        } else {
            for (int64_t m = 0; m < M; ++m)
                std::memset(C + m * ldc, 0, static_cast<size_t>(N) * sizeof(float));
        }
    }

    // ---- clamp bounds from epilogue ----
    float clamp_min = -FLT_MAX;
    float clamp_max =  FLT_MAX;
    bool epilogue_fused = false;
    if (epilogue.type == EpilogueActivateType::Relu) {
        clamp_min = 0.0f;
        epilogue_fused = true;
    } else if (epilogue.type == EpilogueActivateType::None) {
        epilogue_fused = true;
    }

    // ---- workspace (per-panel pack buffers) ----
    const int k_int = static_cast<int>(K);
    std::vector<float> workspace(
        static_cast<size_t>(mr_max_f32 * k_int + nr_max_f32 * k_int));
    float* pack_A = workspace.data();
    float* pack_B = pack_A + mr_max_f32 * k_int;

    // ---- M loop (row panels) ----
    for (int64_t m = 0; m < M; ) {
        const int mr  = choose_mr(M - m);
        const int k_i = k_int;

        // Pack A[m : m+mr, :]
        {
            const float* a_src = transpose_a
                ? A + m              // A is K×M; column m = row m of A^T
                : A + m * lda;       // A is M×K; row m
            tiled_pack_lhs(mr, k_i, pack_A, a_src, static_cast<int>(lda), true);
        }

        // ---- N loop (column panels) ----
        for (int64_t n = 0; n < N; ) {
            const int nr  = choose_nr(N - n);

            // Pack B[:, n : n+nr]
            {
                const float* b_src = transpose_b
                    ? B + n * ldb    // B is N×K; B[n:n+nr, :]
                    : B + n;         // B is K×N; column n
                tiled_pack_rhs(nr, k_i, pack_B, b_src,
                               static_cast<int>(ldb), transpose_b);
            }

            // MMA: C[m:m+mr, n:n+nr] += A_packed × B_packed
            mma_tile(mr, nr, k_i,
                     C + m * ldc + n, static_cast<int>(ldc),
                     pack_A, pack_B, nr,
                     clamp_min, clamp_max);

            // Epilogue (non-fused types)
            if (!epilogue_fused)
                apply_epilogue_block(C + m * ldc + n, ldc, mr, nr, epilogue);

            n += nr;
        }
        m += mr;
    }
}

// ---- 3c. Batch kernel (defined in imatmul.cpp) ----

/// Batch-aware tiled MatMul kernel — conforms to MatMul::Impl::KernelFn.
///
/// Handles N-D batch broadcasting over leading dims, then delegates each
/// 2D sub-matrix to gemm_tiled_f32.
void matmul_tiled_kernel(
    const MatMulAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* workspace);

// ---- 3d. FP16 epilogue block apply ----

/// Apply epilogue (non-clamp types) to an mr×nr output block (fp16).
/// Converts through float for the epilogue function, then back to half_t.
inline void apply_epilogue_block_f16(half_t* C, int64_t ldc, int mr, int nr,
                                      const Epilogue& ep) noexcept {
    for (int m = 0; m < mr; ++m) {
        for (int n = 0; n < nr; ++n) {
            const float val = half_to_float(C[m * ldc + n]);
            C[m * ldc + n] = float_to_half(apply_epilogue(ep, val, n));
        }
    }
}

// ---- 3e. FP16 tiled 2D GEMM ----

/// Tiled float16 GEMM:  C[M][N] = A[M][K] × B[K][N].
///
/// Same decomposition strategy as gemm_tiled_f32 but operates on half_t data.
/// Clamp-based epilogue (Relu/None) is fused into MMA; other types apply
/// per-element via apply_epilogue_block_f16.
///
/// @param C            Output, M × N row-major, ldc stride.
/// @param ldc          Row stride of C.
/// @param A            Input A (half_t).
/// @param lda          Row stride of A.
/// @param B            Input B (half_t).
/// @param ldb          Row stride of B.
/// @param M, N, K      Matrix dimensions.
/// @param transpose_a  If true, A is stored as K×M.
/// @param transpose_b  If true, B is stored as N×K.
/// @param epilogue     Post-processing activation.
/// @param add_to       If true, accumulate into existing C.
/// @param ctx          Compute context (reserved for future parallelism).
inline void gemm_tiled_f16(
    half_t* C, int64_t ldc,
    const half_t* A, int64_t lda,
    const half_t* B, int64_t ldb,
    int64_t M, int64_t N, int64_t K,
    bool transpose_a, bool transpose_b,
    const Epilogue& epilogue,
    bool add_to,
    const ComputeContext& /*ctx*/)
{
    // ---- zero output if overwriting ----
    if (!add_to && M > 0 && N > 0) {
        if (ldc == N) {
            std::memset(C, 0, static_cast<size_t>(M * N) * sizeof(half_t));
        } else {
            for (int64_t m = 0; m < M; ++m)
                std::memset(C + m * ldc, 0, static_cast<size_t>(N) * sizeof(half_t));
        }
    }

    // ---- clamp bounds from epilogue ----
    float clamp_min = -FLT_MAX;
    float clamp_max =  FLT_MAX;
    bool epilogue_fused = false;
    if (epilogue.type == EpilogueActivateType::Relu) {
        clamp_min = 0.0f;
        epilogue_fused = true;
    } else if (epilogue.type == EpilogueActivateType::None) {
        epilogue_fused = true;
    }

    // ---- workspace (per-panel pack buffers) ----
    const int k_int = static_cast<int>(K);
    std::vector<half_t> workspace(
        static_cast<size_t>(mr_max_f16 * k_int + nr_max_f16 * k_int));
    half_t* pack_A = workspace.data();
    half_t* pack_B = pack_A + mr_max_f16 * k_int;

    // ---- M loop (row panels) ----
    for (int64_t m = 0; m < M; ) {
        const int mr  = choose_mr_f16(M - m);
        const int k_i = k_int;

        // Pack A[m : m+mr, :]
        {
            const half_t* a_src = transpose_a
                ? A + m              // A is K×M; column m = row m of A^T
                : A + m * lda;       // A is M×K; row m
            tiled_pack_lhs_f16(mr, k_i, pack_A, a_src, static_cast<int>(lda), true);
        }

        // ---- N loop (column panels) ----
        for (int64_t n = 0; n < N; ) {
            const int nr  = choose_nr_f16(N - n);

            // Pack B[:, n : n+nr]
            {
                const half_t* b_src = transpose_b
                    ? B + n * ldb    // B is N×K; B[n:n+nr, :]
                    : B + n;         // B is K×N; column n
                tiled_pack_rhs_f16(nr, k_i, pack_B, b_src,
                                   static_cast<int>(ldb), transpose_b);
            }

            // MMA: C[m:m+mr, n:n+nr] += A_packed × B_packed
            mma_tile_f16(mr, nr, k_i,
                         C + m * ldc + n, static_cast<int>(ldc),
                         pack_A, pack_B, nr,
                         clamp_min, clamp_max);

            // Epilogue (non-fused types)
            if (!epilogue_fused)
                apply_epilogue_block_f16(C + m * ldc + n, ldc, mr, nr, epilogue);

            n += nr;
        }
        m += mr;
    }
}

// ---- 3f. FP16 batch kernel (defined in imatmul.cpp) ----

/// Batch-aware tiled MatMul kernel for fp16 — conforms to MatMul::Impl::KernelFn.
///
/// Handles N-D batch broadcasting over leading dims, then delegates each
/// 2D sub-matrix to gemm_tiled_f16.
void matmul_tiled_kernel_f16(
    const MatMulAttributes& attrs,
    TensorView& output,
    std::span<const TensorView> inputs,
    const ComputeContext& ctx,
    void* workspace);

}  // namespace nnops::backend::cpu
