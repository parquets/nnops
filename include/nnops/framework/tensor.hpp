#pragma once
/// @file tensor.hpp
/// @brief Tensor — the framework's owning activation buffer.
///
/// The operator layer is deliberately buffer-agnostic: OpBase::compute() takes
/// TensorView, a non-owning view of memory someone else allocated. Tensor is
/// that someone. It owns a dense, 64-byte-aligned block sized by
/// TensorDesc::nbytes() and hands out a TensorView with no row padding, so a
/// value's bytes are exactly its logical elements and anything that reinterprets
/// them — a Reshape in the executor, a memcpy out to the caller — can treat the
/// buffer as one flat run.
///
/// Row pitch stays out of the framework on purpose. Every operator accepts a
/// dense view, and a padded row only pays off when a caller reuses rows across
/// calls, which a graph executor does not do.

#include "nnops/core/tensor_view.hpp"
#include "nnops/core/status.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace nnops::framework {

/// Deleter for the aligned storage; the allocator itself lives in tensor.cpp.
struct AlignedDeleter {
    void operator()(std::byte* p) const noexcept;
};

/// An owning, dense activation (or weight) buffer plus its descriptor.
class Tensor {
public:
    static constexpr size_t kAlignment = 64;

    Tensor() = default;

    Tensor(std::span<const int64_t> shape, DataType dtype,
           TensorLayout layout = TensorLayout::NCHW)
    {
        reset(shape, dtype, layout);
    }

    Tensor(const Tensor& other);
    Tensor& operator=(const Tensor& other);
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;

    /// Reallocate to `shape` (rank <= TensorDesc::kMaxRank); contents are lost.
    Status reset(std::span<const int64_t> shape, DataType dtype,
                 TensorLayout layout = TensorLayout::NCHW);

    /// Reallocate from a descriptor; dims must all be resolved (no -1).
    Status reset(const TensorDesc& desc);

    const TensorDesc& desc() const noexcept { return desc_; }
    int64_t rank() const noexcept { return desc_.rank; }
    const int64_t* shape() const noexcept { return desc_.dims.data(); }
    std::span<const int64_t> shape_span() const noexcept
    {
        return {desc_.dims.data(), static_cast<size_t>(desc_.rank)};
    }
    DataType data_type() const noexcept { return desc_.dtype; }
    TensorLayout layout() const noexcept { return desc_.layout; }
    int64_t numel() const noexcept { return desc_.numel(); }

    /// Bytes of storage; equals TensorDesc::nbytes() (dense, no pitch).
    size_t nbytes() const noexcept { return bytes_; }

    void* data() noexcept { return storage_.get(); }
    const void* data() const noexcept { return storage_.get(); }

    template <typename T>
    T* ptr() noexcept { return static_cast<T*>(data()); }
    template <typename T>
    const T* ptr() const noexcept { return static_cast<const T*>(data()); }

    bool is_empty() const noexcept { return !storage_; }

    /// Dense, non-owning view over the storage.
    TensorView view() const noexcept;

    /// Copy @p bytes in from @p src; the size must equal nbytes().
    Status copy_from(const void* src, size_t bytes);

    /// Copy another tensor; shapes must match.
    Status copy_from(const Tensor& src);

private:
    std::unique_ptr<std::byte[], AlignedDeleter> storage_;
    size_t bytes_ = 0;
    TensorDesc desc_;
};

}  // namespace nnops::framework
