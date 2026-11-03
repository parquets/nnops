#pragma once
/// @file memory_pool.hpp
/// @brief MemoryPool — the CPU backend's internal scratch-memory allocator.
///
/// Operators receive their intermediate ("workspace") memory from the caller
/// (getWorkspaceSize + a pre-allocated buffer); this pool is the internal
/// replacement for the small scratch buffers inside them. Freed blocks are
/// recycled across operators and invocations, avoiding malloc/free churn.
///
/// Design:
///   - Size-class free lists (powers of two, 64 B .. 32 MiB): allocate() rounds
///     up to the next class; deallocate() returns the block to its class.
///   - Every block is 64-byte aligned (matches mma_pack_align).
///   - Each thread keeps a cache of one recycled block per size class, so the
///     "allocate one scratch buffer, free it on return" pattern hits a lock-free
///     fast path. Every thread cache is registered so release_all() can drain it.
///   - Requests above 32 MiB bypass the pool (direct aligned alloc/free).
///   - The central lists are bounded by a retention budget: past it, the largest
///     freed blocks go back to the OS so the idle footprint stays bounded.
///
/// This header is internal to the CPU backend (not part of the public nnops
/// API). The only surface exposed to library consumers is
/// nnops::release_scratch_memory() (see include/nnops/core/backend.hpp), which
/// force-releases all pooled memory.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace nnops::backend::cpu {

class MemoryPool {
public:
    /// Process-wide singleton shared by all CPU operators.
    static MemoryPool& instance();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    /// Allocate @p size bytes, 64-byte aligned: this thread's cache first, then
    /// the shared pool, then the OS. Requests above 32 MiB are direct and not
    /// recycled.
    void* allocate(size_t size);

    /// Return a block from allocate() (nullptr is a no-op). It must come from
    /// this pool; a foreign pointer trips the magic-tag assert in debug builds.
    /// Freeing the same block twice is a programmer error and is not detected.
    void deallocate(void* ptr) noexcept;

    /// Allocate @p count elements of type T, 64-byte aligned.
    template <class T>
    T* allocate_typed(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T)));
    }

    /// Force-release all currently-pooled (free) blocks back to the OS.
    void release_all();

    // ---- statistics ----
    /// Bytes currently handed out to callers.
    size_t live_bytes() const noexcept;
    /// Bytes currently sitting in the free lists (reusable).
    size_t pooled_bytes() const noexcept;
    /// Number of blocks currently sitting in the free lists.
    size_t pooled_blocks() const noexcept;
    /// Total number of allocate() calls.
    uint64_t allocation_count() const noexcept;
    /// Allocate() calls served from the free lists (reuse hits).
    uint64_t reuse_count() const noexcept;

private:
    MemoryPool();
    ~MemoryPool();

    struct Impl;
    Impl* impl_;
};

/// RAII guard for a single pooled allocation: frees the block on destruction.
class PoolPtr {
public:
    PoolPtr() noexcept = default;
    /// Allocate @p size bytes from the process-wide pool.
    explicit PoolPtr(size_t size) : ptr_(MemoryPool::instance().allocate(size)) {}
    ~PoolPtr();

    PoolPtr(const PoolPtr&) = delete;
    PoolPtr& operator=(const PoolPtr&) = delete;

    PoolPtr(PoolPtr&& o) noexcept;
    PoolPtr& operator=(PoolPtr&& o) noexcept;

    void* get() const noexcept { return ptr_; }
    template <class T> T* as() const noexcept { return static_cast<T*>(ptr_); }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /// Free the block (if any) and clear the guard.
    void reset() noexcept;

private:
    void* ptr_ = nullptr;
};

/// Typed, uninitialized scratch array over PoolPtr — the operator-side
/// replacement for std::vector so every per-call request goes through the pool
/// and returns to it on scope exit. Unlike a vector it does not value-initialize
/// its elements; call zero()/assign() when the consumer relies on that.
template <class T>
class PoolArray {
public:
    PoolArray() noexcept = default;
    explicit PoolArray(size_t n) { reset(n); }

    PoolArray(const PoolArray&) = delete;
    PoolArray& operator=(const PoolArray&) = delete;
    PoolArray(PoolArray&& o) noexcept
        : owner_(std::move(o.owner_)), ptr_(o.ptr_), n_(o.n_) {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    PoolArray& operator=(PoolArray&& o) noexcept {
        if (this != &o) {
            owner_ = std::move(o.owner_);
            ptr_ = o.ptr_;
            n_ = o.n_;
            o.ptr_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    T* data() noexcept { return ptr_; }
    const T* data() const noexcept { return ptr_; }
    size_t size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }
    T* begin() noexcept { return ptr_; }
    T* end() noexcept { return ptr_ + n_; }
    const T* begin() const noexcept { return ptr_; }
    const T* end() const noexcept { return ptr_ + n_; }
    T& operator[](size_t i) noexcept { return ptr_[i]; }
    const T& operator[](size_t i) const noexcept { return ptr_[i]; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

    /// Size the array, keeping the block when the size is unchanged.
    bool reset(size_t n) {
        if (n == n_ && (n == 0 || ptr_ != nullptr)) { return true; }
        owner_.reset();
        ptr_ = nullptr;
        n_ = 0;
        if (n == 0) { return true; }
        owner_ = PoolPtr(n * sizeof(T));
        if (!owner_) { return false; }
        ptr_ = owner_.as<T>();
        n_ = n;
        return true;
    }

    /// Size the array and fill every element with @p v.
    void assign(size_t n, T v) { if (reset(n)) { std::fill_n(ptr_, n, v); } }
    /// Size the array and zero every element.
    void zero(size_t n) { assign(n, T{}); }

private:
    PoolPtr owner_;
    T*      ptr_ = nullptr;
    size_t  n_ = 0;
};

}  // namespace nnops::backend::cpu
