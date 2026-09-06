/// @file memory_pool.cpp
/// @brief MemoryPool implementation — size-class free lists + a global mutex.

#include "backend/cpu/common/memory_pool.hpp"
#include "nnops/detail/assert.hpp"

#include <array>
#include <bit>
#include <mutex>
#include <new>

namespace nnops::backend::cpu {

namespace {

constexpr size_t kAlignment = 64;                  // matches GEMM PANEL_ALIGN_BYTES
constexpr size_t kMinClass  = 64;                  // smallest pooled size
constexpr size_t kMaxClass  = 4u * 1024u * 1024u;  // 4 MiB cap
constexpr int    kNumClasses = 17;                 // 64 B .. 4 MiB (powers of two)

constexpr uint32_t kMagic = 0x4E4E4F50u;           // "NNOP" — block sanity tag

// Header stored immediately before the 64-byte-aligned payload. alignas(64)
// rounds sizeof up to 64, so `header + 1` lands the payload on a 64-byte
// boundary (the base is already 64-aligned from the aligned operator new).
struct alignas(kAlignment) PoolHeader {
    PoolHeader* next;    // free-list link; nullptr while the block is live
    size_t      size;    // payload bytes actually allocated (class size if pooled)
    uint32_t    pooled;  // 1 = recyclable size class, 0 = direct oversize
    uint32_t    magic;   // kMagic
};
static_assert(sizeof(PoolHeader) == kAlignment, "header must occupy one cache line");

inline size_t round_up_class(size_t n) noexcept {
    if (n < kMinClass) { n = kMinClass; }
    return std::bit_ceil(n);
}

inline int class_index(size_t cls) noexcept {
    // cls is a power of two in [kMinClass, kMaxClass].
    return static_cast<int>(std::countr_zero(cls) - std::countr_zero(kMinClass));
}

}  // anonymous namespace

struct MemoryPool::Impl {
    std::array<PoolHeader*, kNumClasses> buckets{};

    std::mutex mutex;   // guards everything below
    size_t   live_bytes_    = 0;   // bytes currently handed out
    size_t   pooled_bytes_  = 0;   // bytes in the free lists
    size_t   pooled_blocks_ = 0;   // blocks in the free lists
    uint64_t allocations_ = 0;     // total allocate() calls
    uint64_t reuses_      = 0;     // allocations served from a free list
};

MemoryPool::MemoryPool() : impl_(new Impl) {}

MemoryPool::~MemoryPool() {
    release_all();
    delete impl_;
    impl_ = nullptr;
}

MemoryPool& MemoryPool::instance() {
    static MemoryPool pool;
    return pool;
}

void* MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    ++impl_->allocations_;

    if (size <= kMaxClass) {
        const size_t cls = round_up_class(size);
        const int idx = class_index(cls);

        PoolHeader* h = impl_->buckets[idx];
        if (h != nullptr) {
            impl_->buckets[idx] = h->next;
            impl_->pooled_bytes_ -= cls;
            --impl_->pooled_blocks_;
            ++impl_->reuses_;
            impl_->live_bytes_ += cls;
            h->next = nullptr;
            return h + 1;
        }

        PoolHeader* fresh = static_cast<PoolHeader*>(
            ::operator new(sizeof(PoolHeader) + cls, std::align_val_t{kAlignment}));
        fresh->next = nullptr;
        fresh->size = cls;
        fresh->pooled = 1;
        fresh->magic = kMagic;
        impl_->live_bytes_ += cls;
        return fresh + 1;
    }

    // Oversize: direct allocation, not recycled.
    PoolHeader* h = static_cast<PoolHeader*>(
        ::operator new(sizeof(PoolHeader) + size, std::align_val_t{kAlignment}));
    h->next = nullptr;
    h->size = size;
    h->pooled = 0;
    h->magic = kMagic;
    impl_->live_bytes_ += size;
    return h + 1;
}

void MemoryPool::deallocate(void* ptr) noexcept {
    if (ptr == nullptr) { return; }
    PoolHeader* h = static_cast<PoolHeader*>(ptr) - 1;
    NNOPS_ASSERT(h->magic == kMagic);  // double-free / foreign pointer
    const size_t size = h->size;       // read before h is possibly freed below

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (h->pooled) {
        const int idx = class_index(size);
        h->next = impl_->buckets[idx];
        impl_->buckets[idx] = h;
        impl_->pooled_bytes_ += size;
        ++impl_->pooled_blocks_;
    } else {
        ::operator delete(h, std::align_val_t{kAlignment});
    }
    impl_->live_bytes_ -= size;
}

void MemoryPool::release_all() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (int i = 0; i < kNumClasses; ++i) {
        PoolHeader* h = impl_->buckets[i];
        while (h != nullptr) {
            PoolHeader* next = h->next;
            ::operator delete(h, std::align_val_t{kAlignment});
            h = next;
        }
        impl_->buckets[i] = nullptr;
    }
    impl_->pooled_bytes_ = 0;
    impl_->pooled_blocks_ = 0;
}

size_t MemoryPool::live_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->live_bytes_;
}
size_t MemoryPool::pooled_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->pooled_bytes_;
}
size_t MemoryPool::pooled_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->pooled_blocks_;
}
uint64_t MemoryPool::allocation_count() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->allocations_;
}
uint64_t MemoryPool::reuse_count() const noexcept {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->reuses_;
}

PoolPtr::~PoolPtr() { reset(); }

PoolPtr::PoolPtr(PoolPtr&& o) noexcept : ptr_(o.ptr_) {
    o.ptr_ = nullptr;
}

PoolPtr& PoolPtr::operator=(PoolPtr&& o) noexcept {
    if (this != &o) {
        reset();
        ptr_ = o.ptr_;
        o.ptr_ = nullptr;
    }
    return *this;
}

void PoolPtr::reset() noexcept {
    if (ptr_ != nullptr) {
        MemoryPool::instance().deallocate(ptr_);
    }
    ptr_ = nullptr;
}

}  // namespace nnops::backend::cpu

// ---- public force-release interface (include/nnops/core/backend.hpp) ----

namespace nnops {

void release_scratch_memory() {
    backend::cpu::MemoryPool::instance().release_all();
}

}  // namespace nnops
