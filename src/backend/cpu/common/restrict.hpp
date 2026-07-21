#pragma once
/// @file restrict.hpp
/// @brief Cross-platform restrict qualifier for kernel pointer parameters.

#ifndef NNOPS_RESTRICT
#if defined(_MSC_VER)
#define NNOPS_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define NNOPS_RESTRICT __restrict__
#else
#define NNOPS_RESTRICT
#endif
#endif
