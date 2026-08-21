#pragma once
/// @file imatmul.h
/// @brief Inner matrix multiplication — pack + MMA micro-kernel dispatch.
///
/// All arch-specific pack/mma headers share an identical API in namrspaces
///   nnops::backend::cpu::x86_64   and   nnops::backend::cpu::aarch64.
/// This header picks the right set via NNOPS_ARCH_* and exposes the active
/// namrspace as `arch`, so consumrr code stays clean of #ifdef.

#include "nnops/detail/simd/cpu_features.hpp"  // NNOPS_ARCH_X86_64 / NNOPS_ARCH_AARCH64

// ---- arch-specific includes --------------------------------------------

#ifdef NNOPS_ARCH_X86_64
#include "x86_64/pack_f16.hpp"
#include "x86_64/pack_f32.hpp"
#include "x86_64/mma_direct_f16.hpp"
#include "x86_64/mma_direct_f32.hpp"
#include "x86_64/mma_pack_f16.hpp"
#include "x86_64/mma_pack_f32.hpp"
#elif defined(NNOPS_ARCH_AARCH64)
#include "aarch64/pack_f16.hpp"
#include "aarch64/pack_f32.hpp"
#include "aarch64/mma_direct_f16.hpp"
#include "aarch64/mma_direct_f32.hpp"
#include "aarch64/mma_pack_f16.hpp"
#include "aarch64/mma_pack_f32.hpp"
#endif

#include "nnops/detail/half.hpp"

namespace nnops::backend::cpu {

// ---- namrspace alias ---------------------------------------------------

#ifdef NNOPS_ARCH_X86_64
namespace arch = x86_64;
#elif defined(NNOPS_ARCH_AARCH64)
namespace arch = aarch64;
#endif

// Panel size arrays (largest-first decomposition)
constexpr int MR_F32[3] = {arch::mr_f32[0], arch::mr_f32[1], arch::mr_f32[2]};
constexpr int NR_F32[3] = {arch::nr_f32[0], arch::nr_f32[1], arch::nr_f32[2]};
constexpr int MR_F16[3] = {arch::mr_f16[0], arch::mr_f16[1], arch::mr_f16[2]};
constexpr int NR_F16[3] = {arch::nr_f16[0], arch::nr_f16[1], arch::nr_f16[2]};

constexpr int MR_MAX_F32 = arch::mr_f32[0];
constexpr int NR_MAX_F32 = arch::nr_f32[0];
constexpr int MR_MAX_F16 = arch::mr_f16[0];
constexpr int NR_MAX_F16 = arch::nr_f16[0];


// ---- public API --------------------------------------------------------

void tile_pack_rhs(bool trans, int mc, int kc, float* dst,int ldd, const float* src, int lds, float scale);
void tile_pack_lhs(bool trans, int nc, int kc, float* dst, int ldd, const float* src, int lds, float scale);
void tile_pack_rhs(bool trans, int mc, int kc, half* dst, int ldd, const half* src, int lds, float scale);
void tile_pack_lhs(bool trans, int nc, int kc, half* dst, int ldd, const half* src, int lds, float scale);

void tile_mma_pack(int mc, int nc, int kc, float* c, int ldc, const float* packed_a, int lda, const float* b, int ldb, float clamp_min, float clamp_max);
void tile_mma_direct(int Mc, int nc, int kc, float* c, int ldc, const float* a, int lda, const float* b, int ldb, float clamp_min, float clamp_max);
void tile_mma_pack(int Mc, int nc, int kc, half* c, int ldc, const half* packed_a, int lda, const half* b, int ldb, float clamp_min, float clamp_max);
void tile_mma_direct(int Mc, int nc, int kc, half* c, int ldc, const half* a, int lda, const half* b, int ldb, float clamp_min, float clamp_max);


// single-precision and half-precision GEMM implementations (for matmul.cpp)
template <typename T>
void shgemm_impl(bool trans_a, bool trans_b,
                 bool packed_a, bool packed_b,
                 int Mc, int Nc, int Kc,
                 T* c, int ldc,
                 const T* a, int lda,
                 const T* b, int ldb,
                 float clamp_min, float clamp_max,
                 T* workspace, size_t workspace_bytes);

}  // namespace nnops::backend::cpu

