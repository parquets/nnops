/***************************************************************************************************
 * ptx.h — SM86 HGEMM 底层 PTX 指令封装
 *
 * 适用: __CUDA_ARCH__ >= 800 (SM80/SM86 Ampere)
 * 依赖: CUDA 11+, Clang 14+
 *
 * 每个函数直接对应一条 PTX 指令，不经过任何库的 layout/tensor 抽象层。
 **************************************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// 0. 架构检测
// ============================================================================

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define PTX_SM80_ENABLED 1
#else
#define PTX_SM80_ENABLED 0
#endif

// ============================================================================
// 1. MMA 计算指令 (Tensor Core)
// ============================================================================

// ---------------------------------------------------------------------------
// 2.1  m16n8k8 — f16 * f16 + f16 → f16  (全半精度)
// ---------------------------------------------------------------------------
// 寄存器需求: A=2x uint32, B=1x uint32, C/D=2x uint32
// 每 warp (32 threads) 协作完成 16×8×8 tile 计算

__device__ __forceinline__ void mma_m16n8k8_f16(uint32_t* d0, uint32_t* d1, uint32_t a0,
                                                uint32_t a1, uint32_t b0, uint32_t c0, uint32_t c1)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
                 : "=r"(*d0), "=r"(*d1)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1));
#endif
}

// ---------------------------------------------------------------------------
// 2.2  m16n8k16 — f16 * f16 + f16 → f16  (全半精度, 大K)
// ---------------------------------------------------------------------------
// 寄存器: A=4x uint32, B=2x uint32, C/D=2x uint32

__device__ __forceinline__ void mma_m16n8k16_f16(uint32_t* d0, uint32_t* d1, uint32_t a0,
                                                 uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                                 uint32_t b1, uint32_t c0, uint32_t c1)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                 : "=r"(*d0), "=r"(*d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
#endif
}

// ---------------------------------------------------------------------------
// 2.3  m16n8k8 — f16 * f16 + f32 → f32  (混合精度, 推荐)
// ---------------------------------------------------------------------------
// 寄存器: A=2x uint32, B=1x uint32, C/D=4x float
// FP32 累加，精度最高，最常用

__device__ __forceinline__ void mma_m16n8k8_f32(float* d0, float* d1, float* d2, float* d3,
                                                uint32_t a0, uint32_t a1, uint32_t b0, float c0,
                                                float c1, float c2, float c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                 : "=f"(*d0), "=f"(*d1), "=f"(*d2), "=f"(*d3)
                 : "r"(a0), "r"(a1), "r"(b0), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.4  m16n8k16 — f16 * f16 + f32 → f32  (混合精度, 大K)
// ---------------------------------------------------------------------------
// 寄存器: A=4x uint32, B=2x uint32, C/D=4x float

__device__ __forceinline__ void mma_m16n8k16_f32(float* d0, float* d1, float* d2, float* d3,
                                                 uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                                 uint32_t b0, uint32_t b1, float c0, float c1,
                                                 float c2, float c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(*d0), "=f"(*d1), "=f"(*d2), "=f"(*d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2),
                   "f"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.5  m16n8k16 — bf16 * bf16 + f32 → f32  (混合精度)
// ---------------------------------------------------------------------------
// 寄存器布局与 2.4 完全相同: A=4x uint32, B=2x uint32, C/D=4x float.
// 只有 opcode 的类型后缀不同，所以 16-bit GEMM 的寄存器通路可以共用.

__device__ __forceinline__ void mma_m16n8k16_bf16_f32(float* d0, float* d1, float* d2, float* d3,
                                                      uint32_t a0, uint32_t a1, uint32_t a2,
                                                      uint32_t a3, uint32_t b0, uint32_t b1,
                                                      float c0, float c1, float c2, float c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(*d0), "=f"(*d1), "=f"(*d2), "=f"(*d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2),
                   "f"(c3));
#endif
}

// ============================================================================
// 2. INT8 MMA 计算指令 (s8/u8 → s32, Tensor Core)
// ============================================================================
//
// INT8 MMA 指令格式:
//   mma.sync.aligned.<M>n<N>k<K>.row.col.s32.<Atype>.<Btype>.s32.satfinite
//
// 所有寄存器均为 uint32_t: 每个 reg 打包 4 个 int8 元素.
// 累加器始终为 s32 (4 个 int32 值无需打包, 每个 uint32_t 存 1 个 int32).
// 所有变体均使用 .satfinite — INT32 溢出时钳位而非回绕.
//
// 3 种类型组合:
//   s8.s8  — A signed,   B signed
//   s8.u8  — A signed,   B unsigned
//   u8.s8  — A unsigned, B signed
//
// 3 种 tile 尺寸:
//   m8n8k16  — 小 tile, A=1 reg, B=1 reg, C/D=2 reg
//   m16n8k16 — 中 tile, A=2 reg, B=1 reg, C/D=4 reg
//   m16n8k32 — 大 K,    A=4 reg, B=2 reg, C/D=4 reg

// ---------------------------------------------------------------------------
// 2.1  m8n8k16 — s8.s8 → s32  (A signed, B signed)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m8n8k16_s8s8_satfinite(uint32_t* d0, uint32_t* d1, uint32_t a0,
                                                           uint32_t b0, uint32_t c0, uint32_t c1)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32.satfinite "
                 "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                 : "=r"(*d0), "=r"(*d1)
                 : "r"(a0), "r"(b0), "r"(c0), "r"(c1));
#endif
}

// ---------------------------------------------------------------------------
// 2.2  m8n8k16 — s8.u8 → s32  (A signed, B unsigned)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m8n8k16_s8u8_satfinite(uint32_t* d0, uint32_t* d1, uint32_t a0,
                                                           uint32_t b0, uint32_t c0, uint32_t c1)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.u8.s32.satfinite "
                 "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                 : "=r"(*d0), "=r"(*d1)
                 : "r"(a0), "r"(b0), "r"(c0), "r"(c1));
#endif
}

// ---------------------------------------------------------------------------
// 2.3  m8n8k16 — u8.s8 → s32  (A unsigned, B signed)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m8n8k16_u8s8_satfinite(uint32_t* d0, uint32_t* d1, uint32_t a0,
                                                           uint32_t b0, uint32_t c0, uint32_t c1)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.u8.s8.s32.satfinite "
                 "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                 : "=r"(*d0), "=r"(*d1)
                 : "r"(a0), "r"(b0), "r"(c0), "r"(c1));
#endif
}

// ---------------------------------------------------------------------------
// 2.4  m16n8k16 — s8.s8 → s32  (A signed, B signed)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k16_s8s8_satfinite(uint32_t* d0, uint32_t* d1,
                                                            uint32_t* d2, uint32_t* d3, uint32_t a0,
                                                            uint32_t a1, uint32_t b0, uint32_t c0,
                                                            uint32_t c1, uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32.satfinite "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.5  m16n8k16 — s8.u8 → s32  (A signed, B unsigned)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k16_s8u8_satfinite(uint32_t* d0, uint32_t* d1,
                                                            uint32_t* d2, uint32_t* d3, uint32_t a0,
                                                            uint32_t a1, uint32_t b0, uint32_t c0,
                                                            uint32_t c1, uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32.satfinite "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.6  m16n8k16 — u8.s8 → s32  (A unsigned, B signed)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k16_u8s8_satfinite(uint32_t* d0, uint32_t* d1,
                                                            uint32_t* d2, uint32_t* d3, uint32_t a0,
                                                            uint32_t a1, uint32_t b0, uint32_t c0,
                                                            uint32_t c1, uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.u8.s8.s32.satfinite "
                 "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.7  m16n8k32 — s8.s8 → s32  (A signed, B signed, 大K)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k32_s8s8_satfinite(uint32_t* d0, uint32_t* d1,
                                                            uint32_t* d2, uint32_t* d3, uint32_t a0,
                                                            uint32_t a1, uint32_t a2, uint32_t a3,
                                                            uint32_t b0, uint32_t b1, uint32_t c0,
                                                            uint32_t c1, uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
                   "r"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.8  m16n8k32 — s8.u8 → s32  (A signed, B unsigned, 大K)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k32_s8u8_satfinite(uint32_t* d0, uint32_t* d1,
                                                            uint32_t* d2, uint32_t* d3, uint32_t a0,
                                                            uint32_t a1, uint32_t a2, uint32_t a3,
                                                            uint32_t b0, uint32_t b1, uint32_t c0,
                                                            uint32_t c1, uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32.satfinite "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
                   "r"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.9  m16n8k32 — u8.s8 → s32  (A unsigned, B signed, 大K)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k32_u8s8_satfinite(uint32_t* d0, uint32_t* d1,
                                                            uint32_t* d2, uint32_t* d3, uint32_t a0,
                                                            uint32_t a1, uint32_t a2, uint32_t a3,
                                                            uint32_t b0, uint32_t b1, uint32_t c0,
                                                            uint32_t c1, uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32.satfinite "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
                   "r"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.10 m16n8k32 — s8.s8 → s32  (非饱和)
// ---------------------------------------------------------------------------
// CUTLASS 的 SM80_16x8x32_S32S8S8S32_TN 不带 .satfinite，累加器溢出时按补码回绕。
// 上面 2.7/2.9 是饱和变体，两者在 K·127·127 < 2^31 时逐位相同；这里给的是
// 逐位对应的非饱和形式，供手写 int8 GEMM 使用。

__device__ __forceinline__ void mma_m16n8k32_s8s8(uint32_t* d0, uint32_t* d1, uint32_t* d2,
                                                  uint32_t* d3, uint32_t a0, uint32_t a1,
                                                  uint32_t a2, uint32_t a3, uint32_t b0,
                                                  uint32_t b1, uint32_t c0, uint32_t c1,
                                                  uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
                   "r"(c3));
#endif
}

// ---------------------------------------------------------------------------
// 2.11 m16n8k32 — u8.s8 → s32  (非饱和, A 无符号)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void mma_m16n8k32_u8s8(uint32_t* d0, uint32_t* d1, uint32_t* d2,
                                                  uint32_t* d3, uint32_t a0, uint32_t a1,
                                                  uint32_t a2, uint32_t a3, uint32_t b0,
                                                  uint32_t b1, uint32_t c0, uint32_t c1,
                                                  uint32_t c2, uint32_t c3)
{
#if PTX_SM80_ENABLED
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=r"(*d0), "=r"(*d1), "=r"(*d2), "=r"(*d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
                   "r"(c3));
#endif
}

// ============================================================================
// 3. 数据搬运: Global → Shared Memory (cp.async)
// ============================================================================

/// cp.async — 异步拷贝 Global → Shared, 缓存全层级 (ca = cache all levels)
template <typename T> __device__ __forceinline__ void cp_async_ca(void* smem, const T* gmem)
{
#if PTX_SM80_ENABLED
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr), "l"(gmem),
                 "n"(sizeof(T)));
#endif
}

/// cp.async — 异步拷贝 Global → Shared, 绕过 L1 (cg = cache global only)
template <typename T> __device__ __forceinline__ void cp_async_cg(void* smem, const T* gmem)
{
#if PTX_SM80_ENABLED
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(smem_addr), "l"(gmem),
                 "n"(sizeof(T)));
#endif
}

/// 显式粒度的 cp.async ca 变体 (16B 是最优粒度, 匹配 L2 cache line 128B)
__device__ __forceinline__ void cp_async_ca_16B(void* smem, const void* gmem)
{
#if PTX_SM80_ENABLED
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(smem_addr),
                 "l"(gmem));
#endif
}

// ============================================================================
// 4. cp.async 流水线同步
// ============================================================================

/// 提交一个 cp.async 组 (标记一组异步拷贝的边界)
__device__ __forceinline__ void cp_async_commit_group(void)
{
#if PTX_SM80_ENABLED
    asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}

/// 等待至多有 N 个未完成的 cp.async 组
/// N=0: 等待所有 pending 拷贝完成
/// N=1: 允许 1 组未完成 (典型软件流水线模式)
///
/// 本地改：N 作为模板参数。PTX 的 "n" 约束要求立即数，而 nvcc 会编译每一个
/// 非模板 __device__ 函数（即使未被调用），运行期 int 参数会让本头无法编译。
template <int N> __device__ __forceinline__ void cp_async_wait_group()
{
#if PTX_SM80_ENABLED
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
#endif
}

/// 等待所有 pending cp.async 完成
__device__ __forceinline__ void cp_async_wait_all(void)
{
#if PTX_SM80_ENABLED
    asm volatile("cp.async.wait_all;\n" ::: "memory");
#endif
}

// ============================================================================
// 5. Shared Memory → 寄存器 (ldmatrix / ld.shared)
// ============================================================================

/// ldmatrix.x2 — 每线程加载 2 个 uint32 寄存器
__device__ __forceinline__ void ldmatrix_x2(uint32_t* r0, uint32_t* r1, const void* smem)
{
#if PTX_SM80_ENABLED
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(*r0), "=r"(*r1)
                 : "r"(smem_addr));
#endif
}

/// ldmatrix.x4 — 每线程加载 4 个 uint32 寄存器
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3,
                                            const void* smem)
{
#if PTX_SM80_ENABLED
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3)
                 : "r"(smem_addr));
#endif
}

/// ldmatrix.x2.trans — 加载时转置
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* r0, uint32_t* r1, const void* smem)
{
#if PTX_SM80_ENABLED
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(*r0), "=r"(*r1)
                 : "r"(smem_addr));
#endif
}

/// ldmatrix.x4.trans — 加载时转置
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                                                  uint32_t* r3, const void* smem)
{
#if PTX_SM80_ENABLED
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3)
                 : "r"(smem_addr));
#endif
}

/// 共享内存地址转成 32 位偏移 (cvta 一次)。给了地址就不再走指针的 64 位运算：
/// 内层循环里每条 ldmatrix 只需 `base + 编译期常量`，ptxas 通常把这个加直接折进
/// LDS 的立即数偏移。
__device__ __forceinline__ uint32_t smem_addr_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

/// 给定 32 位共享地址的 ldmatrix.x4 / x2 (省掉每次调用的 cvta)。
__device__ __forceinline__ void ldmatrix_x4_at(uint32_t* r0, uint32_t* r1, uint32_t* r2,
                                               uint32_t* r3, uint32_t addr)
{
#if PTX_SM80_ENABLED
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3)
                 : "r"(addr));
#endif
}

__device__ __forceinline__ void ldmatrix_x2_at(uint32_t* r0, uint32_t* r1, uint32_t addr)
{
#if PTX_SM80_ENABLED
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(*r0), "=r"(*r1)
                 : "r"(addr));
#endif
}

/// ld.shared.b32 — 通用 32-bit smem read
__device__ __forceinline__ void ld_shared_32b(uint32_t* dst, const void* smem)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(*dst) : "r"(smem_addr));
}

/// ld.shared.v2.b32 — 通用 64-bit smem read
__device__ __forceinline__ void ld_shared_64b(uint32_t* dst0, uint32_t* dst1, const void* smem)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(*dst0), "=r"(*dst1) : "r"(smem_addr));
}

/// ld.shared.v4.b32 — 通用 128-bit smem read
__device__ __forceinline__ void ld_shared_128b(uint32_t* dst0, uint32_t* dst1, uint32_t* dst2,
                                               uint32_t* dst3, const void* smem)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(*dst0), "=r"(*dst1), "=r"(*dst2), "=r"(*dst3)
                 : "r"(smem_addr));
}

// ============================================================================
// 6. 寄存器 → Shared Memory (st.shared, Epilogue 用)
// ============================================================================

/// st.shared.b32 — 通用 32-bit smem write
__device__ __forceinline__ void st_shared_32b(void* smem, uint32_t val)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(smem_addr), "r"(val));
}

/// st.shared.v2.b32 — 通用 64-bit smem write
__device__ __forceinline__ void st_shared_64b(void* smem, uint32_t v0, uint32_t v1)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};\n" ::"r"(smem_addr), "r"(v0), "r"(v1));
}

/// st.shared.v4.b32 — 通用 128-bit smem write
__device__ __forceinline__ void st_shared_128b(void* smem, uint32_t v0, uint32_t v1, uint32_t v2,
                                               uint32_t v3)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n" ::"r"(smem_addr), "r"(v0), "r"(v1),
                 "r"(v2), "r"(v3));
}

// ============================================================================
// 7. 同步原语
// ============================================================================

/// Warp 内同步 (bar.warp.sync 0xffffffff)
/// mma.sync / ldmatrix.sync 自身隐含 warp 同步，cp.async / ld.shared 不隐含
__device__ __forceinline__ void sync_warp(void)
{
    asm volatile("bar.warp.sync 0xffffffff;\n" ::: "memory");
}

/// Block 内同步 (__syncthreads)
__device__ __forceinline__ void sync_block(void) { __syncthreads(); }
