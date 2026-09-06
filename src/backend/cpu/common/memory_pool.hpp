#pragma once
/// @file memory_pool.hpp
/// @brief MemoryPool — the CPU backend's internal scratch-memory allocator.
///
/// Operators currently receive their intermediate ("workspace") memory from the
/// caller (getWorkspaceSize + a pre-allocated buffer). This pool is the internal
/// replacement: operators allocate/free small scratch buffers through a shared
/// process-wide singleton, and freed blocks are recycled across operators and
/// invocations, avoiding repeated malloc/free churn.
///
/// Design:
///   - Size-class free lists (powers of two, 64 B .. 4 MiB). allocate() rounds
///     the request up to the next class; deallocate() returns the block to its
///     class bucket for reuse by the next request of a matching size.
///   - Every block is 64-byte aligned (matches the GEMM PANEL_ALIGN_BYTES).
///   - A single mutex serializes pool operations. This is deliberate: scratch
///     allocation happens once per operator call, not per element, so the lock
///     cost is negligible even under parallel operator execution.
///   - Requests above 4 MiB bypass the pool and use a direct aligned alloc/free.
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

    /// Allocate @p size bytes, 64-byte aligned. Reuses a pooled block when one
    /// is available; otherwise allocates fresh from the OS. Requests larger than
    /// the max class size (4 MiB) are allocated directly and not recycled.
    void* allocate(size_t size);

    /// Return a block previously obtained from allocate(). nullptr is a no-op.
    /// The block must have come from this pool; passing a foreign pointer or
    /// freeing twice is a programmer error (asserted in debug builds).
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
