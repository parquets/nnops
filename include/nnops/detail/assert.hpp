#pragma once
/// @file assert.hpp
/// @brief Minimal assertion utility for nnops.
///
/// Two levels, both compiled out under NDEBUG:
///   - NNOPS_ASSERT(_MSG) aborts. For contract violations.
///   - NNOPS_WARN_MSG     only prints. For configurations that are legal but
///                        usually a mistake — an abort there would be wrong.
///
/// NNOPS_CHECK_MSG is the NDEBUG-proof subset: use it where continuing would
/// corrupt memory or silently produce wrong output (an unsupported dtype on a
/// path that has no fallback, a missing backend kernel).

#include <cstdio>
#include <cstdlib>

// Always-on contract check — for preconditions whose violation would read or
// write the wrong memory rather than merely be a caller bug.
#define NNOPS_CHECK_MSG(cond, msg)                                       \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "NNOPS_CHECK failed: %s\n  msg: %s\n  at %s:%d\n", \
                   #cond, msg, __FILE__, __LINE__);                      \
      std::abort();                                                      \
    }                                                                    \
  } while (0)

#define NNOPS_CHECK(cond) NNOPS_CHECK_MSG(cond, "")

#if defined(NDEBUG)
#define NNOPS_ASSERT(cond) ((void)0)
#define NNOPS_ASSERT_MSG(cond, msg) ((void)0)
#define NNOPS_WARN_MSG(cond, msg) ((void)0)
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

// Non-fatal counterpart of NNOPS_ASSERT_MSG: same "condition that ought to
// hold" convention, but a false condition only prints — the call proceeds.
#define NNOPS_WARN_MSG(cond, msg)                                     \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "NNOPS_WARN: %s\n  msg: %s\n  at %s:%d\n",  \
                   #cond, msg, __FILE__, __LINE__);                   \
    }                                                                 \
  } while (0)
#endif
