#pragma once
/// @file assert.hpp
/// @brief Minimal assertion utility for nnops.

#include <cstdio>
#include <cstdlib>

#if defined(NDEBUG)
#define NNOPS_ASSERT(cond) ((void)0)
#define NNOPS_ASSERT_MSG(cond, msg) ((void)0)
#else
#define NNOPS_ASSERT(cond)                                            \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "NNOPS_ASSERT failed: %s\n  at %s:%d\n",  \
                   #cond, __FILE__, __LINE__);                        \
      std::abort();                                                   \
    }                                                                 \
  } while (0)

#define NNOPS_ASSERT_MSG(cond, msg)                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "NNOPS_ASSERT failed: %s\n  msg: %s\n  at %s:%d\n", \
                   #cond, msg, __FILE__, __LINE__);                   \
      std::abort();                                                   \
    }                                                                 \
  } while (0)
#endif
