#pragma once
/// @file validate.hpp
/// @brief Argument-check helpers shared by the operator dispatch layer.
///
/// These replace the NNOPS_ASSERT calls that used to guard caller input: an
/// assert is compiled out under NDEBUG, which turns a malformed call into an
/// out-of-bounds access. A returned Status stays live in every build.
///
/// Two levels, matching the two entry points:
///   check_counts / check_views — tensor counts (and, for views, that each one
///     holds data). validate() and compute() both run these.
///   everything else — descriptor-level rules that validate() owns.

#include "nnops/core/backend.hpp"
#include "nnops/core/status.hpp"
#include "nnops/core/tensor_view.hpp"
#include "nnops/detail/bf16.hpp"  // bf16_{cpu,cuda}_enabled (per-backend gate)

#include <cstddef>
#include <span>

namespace nnops::detail {

/// Evaluate `expr` (returning Status); on error, return it from the caller.
#define NNOPS_TRY(expr)                                \
    do {                                               \
        const ::nnops::Status nnops_st_ = (expr);      \
        if (!::nnops::is_ok(nnops_st_)) {              \
            return nnops_st_;                          \
        }                                              \
    } while (0)

constexpr Status require(bool cond,
                         Status code = Status::invalid_argument) noexcept
{
    return cond ? Status::ok : code;
}

/// Tensor-count check. n_in_max == size_t(-1) means "at least n_in_min".
constexpr Status check_counts(size_t n_in, size_t n_out,
                              size_t n_in_min, size_t n_in_max = size_t(-1),
                              size_t n_out_min = 1) noexcept
{
    if (n_in < n_in_min) { return Status::invalid_argument; }
    if (n_in_max != size_t(-1) && n_in > n_in_max) {
        return Status::invalid_argument;
    }
    if (n_out < n_out_min) { return Status::invalid_argument; }
    return Status::ok;
}

/// Whether `be` carries bf16 kernels: the CPU on AArch64, CUDA wherever the
/// backend is built. Every other backend, and every other build, says no.
///
/// Both operands are compile-time constants, so a build that carries neither
/// folds this to `false` and a build with one drops the other comparison; the
/// switch form this replaced survived as a jump table either way.
constexpr bool backend_has_bf16(Backend be) noexcept
{
    return (bf16_cpu_enabled && be == Backend::CPU) ||
           (bf16_cuda_enabled && be == Backend::CUDA);
}

/// Counts, plus every participating view must hold data and carry a dtype the
/// *selected backend* implements. bf16 is the only dtype whose availability is
/// per backend, so a bf16 view is refused here — the one call every operator's
/// compute() makes — and the call returns Status::unsupported instead of
/// reaching a dispatch that has no bf16 case. An operator that lists no dtype in
/// validate() (the movement ops) is covered by the same gate.
inline Status check_views(Backend be,
                          std::span<TensorView> outputs,
                          std::span<const TensorView> inputs,
                          size_t n_in_min, size_t n_in_max = size_t(-1),
                          size_t n_out = 1) noexcept
{
    NNOPS_TRY(check_counts(inputs.size(), outputs.size(), n_in_min, n_in_max, n_out));
    // A build with no bf16 anywhere compiles the dtype scan out entirely, and
    // with it this flag.
    [[maybe_unused]] const bool bf16_ok = backend_has_bf16(be);
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].is_empty()) { return Status::invalid_argument; }
        if constexpr (bf16_enabled) {
            if (!bf16_ok && inputs[i].data_type() == DataType::bf16) {
                return Status::unsupported;
            }
        }
    }
    for (size_t i = 0; i < n_out; ++i) {
        if (outputs[i].is_empty()) { return Status::invalid_argument; }
        if constexpr (bf16_enabled) {
            if (!bf16_ok && outputs[i].data_type() == DataType::bf16) {
                return Status::unsupported;
            }
        }
    }
    return Status::ok;
}

/// bf16 is a float type: it is admitted wherever f16 is admitted, on a backend
/// that carries bf16. The backend is what decides the last part — an x86-64 CPU
/// has no bf16 kernel in any build — so the gate is a property of the call and
/// takes the operator's backend alongside the dtype.
template <DataType... Ds>
constexpr bool dtype_in(Backend be, DataType d) noexcept
{
    if constexpr (bf16_enabled) {
        if (d == DataType::bf16) {
            return backend_has_bf16(be) && ((Ds == DataType::f16) || ...);
        }
    }
    return ((d == Ds) || ...);
}

/// Rank must lie in [lo, hi]; hi < 0 means unbounded.
constexpr Status check_rank(int64_t rank, int64_t lo, int64_t hi = -1) noexcept
{
    return (rank >= lo && (hi < 0 || rank <= hi)) ? Status::ok
                                                  : Status::invalid_argument;
}

}  // namespace nnops::detail
