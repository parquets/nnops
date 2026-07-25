/// @file matmul.cpp
/// @brief Tiled matrix multiplication kernel — NKM loop with imatmul pack + MMA.
///
/// Determines tile sizes Mc, Nc from the L2 cache constraint:
///   (mr × Kc + Nc × Kc + mr × Nc) × 2 × elem_size < L2_SIZE
/// where mr is the micro-kernel M panel (not Mc — only the active mr panel
/// is in the L2 working set during MMA).

#include "matmul.h"
#include "imatmul.h"
#include "nnops/detail/half.hpp"
#include "nnops/detail/simd/cpu_features.hpp"
#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <cstring>
#include <limits>

// Forward-declare reference kernel for batched fallback.
namespace nnops::backend::cpu::reference {
extern void matmul_ref(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       const ComputeContext& ctx,
                       void* workspace);
}

// =========================================================================
//  Architecture-conditional: f16 pointer types, panel sizes
// =========================================================================
// Duplicates the arch block from imatmul.cpp — these are ISA facts,
// not implementation details of imatmul.

#if defined(NNOPS_ARCH_X86_64)
  #define F16_PTR  half*
  #define F16_CPTR const half*
  // f32
  #define F32_MR0 6
  #define F32_MR1 4
  #define F32_NR0 16
  #define F32_NR1 8
  // f16
  #define F16_MR0 6
  #define F16_MR1 4
  #define F16_NR0 16
  #define F16_NR1 8
#elif defined(NNOPS_ARCH_AARCH64)
  #define F16_PTR  float16_t*
  #define F16_CPTR const float16_t*
  // f32
  #define F32_MR0 8
  #define F32_MR1 4
  #define F32_NR0 12
  #define F32_NR1 4
  // f16
  #define F16_MR0 8
  #define F16_MR1 4
  #define F16_NR0 16
  #define F16_NR1 8
#endif

#define F32_MR2 1
#define F32_NR2 1
#define F16_MR2 1
#define F16_NR2 1

namespace nnops::backend::cpu {
namespace {

// =========================================================================
//  Tiling constants
// =========================================================================

constexpr int KC_F32 = 128;
constexpr int KC_F16 = 256;
constexpr int MC_TARGET = 192;  // 192/6=32 (x86), 192/8=24 (aarch64)

// Panel size arrays (largest-first decomposition)
constexpr int MR_F32[3] = {F32_MR0, F32_MR1, F32_MR2};
constexpr int NR_F32[3] = {F32_NR0, F32_NR1, F32_NR2};
constexpr int MR_F16[3] = {F16_MR0, F16_MR1, F16_MR2};
constexpr int NR_F16[3] = {F16_NR0, F16_NR1, F16_NR2};

constexpr int MR_MAX_F32 = F32_MR0;
constexpr int NR_MAX_F32 = F32_NR0;
constexpr int MR_MAX_F16 = F16_MR0;
constexpr int NR_MAX_F16 = F16_NR0;

// =========================================================================
//  Nc from L2 constraint
// =========================================================================
// From: (mr × Kc + Nc × Kc + mr × Nc) × 2 × elem_size < L2_SIZE
// → Nc < (L2_SIZE / (2 × elem_size) - mr × Kc) / (Kc + mr)

inline int compute_nc_f32(int mr_max, size_t l2_size) noexcept {
    int rhs   = static_cast<int>(l2_size / 8) - mr_max * KC_F32;
    int denom = KC_F32 + mr_max;
    return (rhs / denom) - 1;  // -1 for safety margin
}

inline int compute_nc_f16(int mr_max, size_t l2_size) noexcept {
    int rhs   = static_cast<int>(l2_size / 4) - mr_max * KC_F16;
    int denom = KC_F16 + mr_max;
    return (rhs / denom) - 1;
}

inline int round_down_nc(int nc, int nr_max) noexcept {
    return (nc / nr_max) * nr_max;
}

// =========================================================================
//  Workspace
// =========================================================================

inline size_t workspace_bytes(int mc, int nc, int kc, size_t elem) noexcept {
    return static_cast<size_t>(mc + nc) * static_cast<size_t>(kc) * elem;
}

// =========================================================================
//  matmul_kernel_f32
// =========================================================================

void matmul_kernel_f32(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       void* workspace)
{

}

// =========================================================================
//  matmul_kernel_f16
// =========================================================================

void matmul_kernel_f16(const MatMulAttributes& attrs,
                       TensorView& output,
                       std::span<const TensorView> inputs,
                       void* workspace)
{

}

}  // anonymous namespace

// =========================================================================
//  Public API
// =========================================================================

size_t matmul_get_workspace_size(const MatMulAttributes& attrs,
                                 const TensorDesc& a_desc,
                                 const TensorDesc& b_desc,
                                 const TensorDesc& /*c_desc*/)
{
    int64_t a_rank = a_desc.rank;
    int64_t b_rank = b_desc.rank;
    // M = rows of A in its logical (non-transposed) layout
    // N = cols of B in its logical (non-transposed) layout
    int64_t M = attrs.transpose_a ? a_desc.dims[static_cast<size_t>(a_rank - 1)]
                                  : a_desc.dims[static_cast<size_t>(a_rank - 2)];
    int64_t N = attrs.transpose_b ? b_desc.dims[static_cast<size_t>(b_rank - 2)]
                                  : b_desc.dims[static_cast<size_t>(b_rank - 1)];

    size_t l2_size = simd::CpuFeatures::get().l2_cache_size();

    if (a_desc.dtype == DataType::f16) {
        int nc = round_down_nc(compute_nc_f16(MR_MAX_F16, l2_size), NR_MAX_F16);
        int mc = std::min(MC_TARGET, static_cast<int>(M));
        nc = std::min(nc, static_cast<int>(N));
        return workspace_bytes(mc, nc, KC_F16, 2);
    } else {
        int nc = round_down_nc(compute_nc_f32(MR_MAX_F32, l2_size), NR_MAX_F32);
        int mc = std::min(MC_TARGET, static_cast<int>(M));
        nc = std::min(nc, static_cast<int>(N));
        return workspace_bytes(mc, nc, KC_F32, 4);
    }
}

void matmul_kernel(const MatMulAttributes& attrs,
                   TensorView& output,
                   std::span<const TensorView> inputs,
                   const ComputeContext& ctx,
                   void* workspace)
{
    const auto& a = inputs[0];
    const auto& b = inputs[1];

    NNOPS_ASSERT(a.rank() >= 2);
    NNOPS_ASSERT(b.rank() >= 2);
    NNOPS_ASSERT(inputs.size() >= 2 && inputs.size() <= 3);

    // 2D only for now — batched falls through to reference
    if (a.rank() != 2 || b.rank() != 2) {
        reference::matmul_ref(attrs, output, inputs, ctx, workspace);
        return;
    }

    if (a.data_type() == DataType::f16)
        matmul_kernel_f16(attrs, output, inputs, workspace);
    else
        matmul_kernel_f32(attrs, output, inputs, workspace);
}

}  // namespace nnops::backend::cpu

// =========================================================================
//  Clean up file-scoped macros
// =========================================================================
#undef F16_PTR
#undef F16_CPTR
#undef F32_MR0
#undef F32_MR1
#undef F32_MR2
#undef F32_NR0
#undef F32_NR1
#undef F32_NR2
#undef F16_MR0
#undef F16_MR1
#undef F16_MR2
#undef F16_NR0
#undef F16_NR1
#undef F16_NR2
