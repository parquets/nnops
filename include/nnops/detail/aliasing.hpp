#pragma once
/// @file aliasing.hpp
/// @brief The descriptor-level test behind OpBase::supports_inplace().
///
/// supports_inplace() answers a plan-time question: may outputs[0] be the same
/// storage as an input? The answer is a property of the operator, its backend
/// and the element geometry — never of the addresses, which a TensorDesc does
/// not carry. The shared condition is that the input and the output describe the
/// same elements in the same order, so an element-wise (or row-wise) kernel's
/// index mapping is the identity and every store lands on an element already
/// read.
///
/// Two obligations stay with the caller because descriptors cannot see them:
/// both views must be built with the same pitch, and only an input whose
/// descriptor matches may be aliased — a partial overlap is never covered.

#include "nnops/core/tensor_view.hpp"

#include <cstddef>
#include <span>

namespace nnops::detail {

/// Same rank, extents, dtype and layout — and the same row stride when both
/// descriptors state one. That is exactly when a store at (r, c) into @p b
/// reads what @p a held at (r, c).
inline bool same_element_geometry(const TensorDesc& a, const TensorDesc& b) noexcept
{
    if (a.rank != b.rank || a.dtype != b.dtype || a.layout != b.layout) {
        return false;
    }
    for (int64_t i = 0; i < a.rank; ++i) {
        if (a.dims[static_cast<size_t>(i)] != b.dims[static_cast<size_t>(i)]) {
            return false;
        }
    }
    return a.row_stride_elems == 0 || b.row_stride_elems == 0 ||
           a.row_stride_elems == b.row_stride_elems;
}

/// The shared body of the supports_inplace() overrides: true when the single
/// output may take the storage of an input that describes the same geometry.
inline bool may_alias_any_input(std::span<const TensorDesc> inputs,
                                std::span<const TensorDesc> outputs) noexcept
{
    if (outputs.empty()) {
        return false;
    }
    for (const TensorDesc& in : inputs) {
        if (same_element_geometry(in, outputs[0])) {
            return true;
        }
    }
    return false;
}

} // namespace nnops::detail

/// Defines the in-place contract for an operator whose CPU kernels are
/// element-wise (or row-wise): read every element before storing it, so any
/// input that describes the output's element geometry may share its storage.
#define NNOPS_INPLACE_ELEMENTWISE(Op)                                     \
    bool Op::supports_inplace(std::span<const TensorDesc> inputs,         \
                              std::span<const TensorDesc> outputs) const  \
    {                                                                     \
        return backend_ == Backend::CPU &&                                \
               ::nnops::detail::may_alias_any_input(inputs, outputs);     \
    }
