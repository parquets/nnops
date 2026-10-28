#pragma once
/// @file status.hpp
/// @brief Status — the error code returned by operators.
///
/// Every fallible entry point returns a Status. Caller mistakes (wrong shape,
/// dtype, tensor count, missing workspace) come back as a code; the library
/// only aborts on an internal invariant it has already checked — see
/// detail/assert.hpp for that boundary.
///
/// Why a code and not an exception: nnops is dropped into host engines that
/// may be built with exceptions off, and a status is a value type that
/// crosses any ABI. Code that ignores the result still compiles, so adopting
/// the return value is opt-in.

#include <cstdint>

namespace nnops {

enum class Status : uint8_t {
    ok = 0,
    /// The call is malformed: wrong tensor count, empty view, rank/dtype/
    /// layout the operator's contract does not allow, bad attribute value.
    invalid_argument,
    /// The call is well-formed but this backend has no implementation for it
    /// (dtype, layout, quantization granularity, fusions such as add_to).
    unsupported,
    /// Required state is absent: a workspace the operator asked for, a
    /// weight packed by prepackWeights().
    not_initialized,
    out_of_memory,
    /// A library invariant failed. Reaching this is a bug in nnops.
    internal,
};

constexpr bool is_ok(Status s) noexcept { return s == Status::ok; }
constexpr bool is_error(Status s) noexcept { return s != Status::ok; }

inline const char* to_string(Status s) noexcept
{
    switch (s) {
    case Status::ok:               return "ok";
    case Status::invalid_argument: return "invalid_argument";
    case Status::unsupported:      return "unsupported";
    case Status::not_initialized:  return "not_initialized";
    case Status::out_of_memory:    return "out_of_memory";
    case Status::internal:         return "internal";
    }
    return "unknown";
}

}  // namespace nnops
