#pragma once
/// @file simd.hpp
/// @brief Architecture detection and SIMD abstraction entry point.
///
/// This is the backward-compatible include for code that uses the original
/// arch-detection macros. For the full SIMD vector API, include the
/// subdirectory header directly:
///   #include "nnops/detail/simd/simd.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64)
  #define NNOPS_ARCH_X86_64 1
#elif defined(__aarch64__) || defined(_M_ARM64)
  #define NNOPS_ARCH_AARCH64 1
#endif

// Include the full SIMD abstraction layer
#include "nnops/detail/simd/simd.hpp"
