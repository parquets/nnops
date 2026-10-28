#pragma once
/// @file assert.hpp
/// @brief Minimal assertion utility for nnops.
///
/// These are for *internal invariants* — a state the library's own code
/// established and then relies on. Anything a caller can get wrong is reported
/// with a returned Status instead (see nnops/core/status.hpp and
/// nnops/detail/validate.hpp): an assert is compiled out under NDEBUG, which
/// would turn a malformed call into an out-of-bounds access, and an abort takes
/// the host process down with it.
///
/// Two levels, both compiled out under NDEBUG:
///   - NNOPS_ASSERT(_MSG) aborts. For internal invariants.
///   - NNOPS_WARN_MSG     only prints. For configurations that are legal but
///                        usually a mistake — an abort there would be wrong.
///
/// NNOPS_CHECK_MSG is the NDEBUG-proof form of NNOPS_ASSERT: it stays live in
/// release builds. Nothing in src/ uses it any more — the caller-facing checks
/// it used to guard now return Status — so reserve it for an invariant whose
/// violation in release would corrupt memory and cannot be reported.

#include <cstdio>
#include <cstdlib>

// Always-on internal-invariant check — for a state whose violation would read
// or write the wrong memory in release, where NNOPS_ASSERT no longer exists.
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
