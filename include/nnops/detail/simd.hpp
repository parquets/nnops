#pragma once
/// @file simd.hpp
/// @brief Architecture detection macros for CPU-specific optimizations.

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
  #define NNOPS_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define NNOPS_ARCH_AARCH64 1
#endif
