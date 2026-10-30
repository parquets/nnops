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
///   - Size-class free lists (powers of two, 64 B .. 16 MiB): allocate() rounds
///     up to the next class; deallocate() returns the block to its class.
///   - Every block is 64-byte aligned (matches mma_pack_align).
///   - Each thread keeps a cache of one recycled block per size class, so the
///     "allocate one scratch buffer, free it on return" pattern hits a lock-free
///     fast path. Every thread cache is registered so release_all() can drain it.
///   - Requests above 16 MiB bypass the pool (direct aligned alloc/free).
///   - The central lists are bounded by a retention budget: past it, the largest
///     freed blocks go back to the OS so the idle footprint stays bounded.
///
/// This header is internal to the CPU backend (not part of the public nnops
/// API). The only surface exposed to library consumers is
/// nnops::release_scratch_memory() (see include/nnops/core/backend.hpp), which
/// force-releases all pooled memory.

#include <cstddef>
#include <cstdint>

namespace nnops::backend::cpu {

class MemoryPool {
public:
    /// Process-wide singleton shared by all CPU operators.
    static MemoryPool& instance();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    /// Allocate @p size bytes, 64-byte aligned: this thread's cache first, then
    /// the shared pool, then the OS. Requests above 16 MiB are direct and not
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

}  // namespace nnops::backend::cpu
