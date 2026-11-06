/// @file tensor.cpp
/// @brief Tensor storage: the aligned allocator, dense views, and copies.
///
/// The block is allocated once per reset() at 64 bytes — the widest cache line
/// the kernels touch — because every operator loads through intrinsics that
/// prefer aligned addresses and the channel-last routes take whole rows. The
/// size is TensorDesc::nbytes() (elements, no row padding) since the framework
/// never hands an operator a pitched view.

#include "nnops/framework/tensor.hpp"

#include <cstdlib>
#include <cstring>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace nnops::framework {

namespace {

std::byte* aligned_allocate(size_t bytes) noexcept
{
    if (bytes == 0) { return nullptr; }
#if defined(_MSC_VER)
    return static_cast<std::byte*>(_aligned_malloc(bytes, Tensor::kAlignment));
#else
    // aligned_alloc wants a size that is a multiple of the alignment.
    const size_t rounded = (bytes + Tensor::kAlignment - 1) / Tensor::kAlignment * Tensor::kAlignment;
    return static_cast<std::byte*>(std::aligned_alloc(Tensor::kAlignment, rounded));
#endif
}

void aligned_release(std::byte* p) noexcept
{
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

bool dims_resolved(const TensorDesc& desc) noexcept
{
    for (int64_t i = 0; i < desc.rank; ++i) {
        if (desc.dims[static_cast<size_t>(i)] < 0) { return false; }
    }
    return true;
}

}  // namespace

void AlignedDeleter::operator()(std::byte* p) const noexcept
{
    aligned_release(p);
}

Tensor::Tensor(const Tensor& other)
    : storage_(nullptr, AlignedDeleter{}), bytes_(0), desc_(other.desc_)
{
    if (!other.is_empty()) {
        const Status st = reset(other.desc_);
        if (is_ok(st)) {
            std::memcpy(storage_.get(), other.storage_.get(), other.bytes_);
        }
    }
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this == &other) { return *this; }
    storage_.reset();
    bytes_ = 0;
    desc_ = other.desc_;
    if (!other.is_empty()) {
        if (is_ok(reset(other.desc_))) {
            std::memcpy(storage_.get(), other.storage_.get(), other.bytes_);
        }
    }
    return *this;
}

Status Tensor::reset(std::span<const int64_t> shape, DataType dtype, TensorLayout layout)
{
    if (shape.size() > static_cast<size_t>(TensorDesc::kMaxRank)) {
        return Status::invalid_argument;
    }
    TensorDesc desc;
    desc.rank = static_cast<int64_t>(shape.size());
    desc.dims = shape;
    desc.dtype = dtype;
    desc.layout = layout;
    return reset(desc);
}

Status Tensor::reset(const TensorDesc& desc)
{
    if (desc.rank > TensorDesc::kMaxRank || !dims_resolved(desc)) {
        return Status::invalid_argument;
    }
    const size_t bytes = desc.nbytes();
    if (bytes == 0) {
        // A zero-extent tensor is legal (an empty batch): keep the descriptor
        // and hand out a null pointer, which is_empty() reports.
        storage_.reset();
        bytes_ = 0;
        desc_ = desc;
        return Status::ok;
    }
    std::byte* block = aligned_allocate(bytes);
    if (block == nullptr) { return Status::out_of_memory; }
    storage_.reset(block);
    bytes_ = bytes;
    desc_ = desc;
    return Status::ok;
}

TensorView Tensor::view() const noexcept
{
    if (desc_.rank == 0) {
        return TensorView(std::span<const int64_t>(), desc_.dtype,
                          storage_.get(), desc_.layout);
    }
    return TensorView(std::span<const int64_t>(desc_.dims.data(),
                                               static_cast<size_t>(desc_.rank)),
                      desc_.dtype, storage_.get(), desc_.layout);
}

Status Tensor::copy_from(const void* src, size_t bytes)
{
    if (src == nullptr || bytes != bytes_) { return Status::invalid_argument; }
    if (bytes != 0) { std::memcpy(storage_.get(), src, bytes); }
    return Status::ok;
}

Status Tensor::copy_from(const Tensor& src)
{
    if (src.nbytes() != bytes_) { return Status::invalid_argument; }
    return copy_from(src.data(), src.nbytes());
}

}  // namespace nnops::framework
