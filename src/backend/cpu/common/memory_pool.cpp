/// @file memory_pool.cpp
/// @brief MemoryPool implementation — per-thread caches over size-class free
///        lists, a bounded retention budget, and a thread-cache registry so
///        release_all() can drain every thread.

#include "backend/cpu/common/memory_pool.hpp"
#include "nnops/detail/assert.hpp"

#include <array>
#include <atomic>
#include <bit>
#include <mutex>
#include <new>
#include <vector>

namespace nnops::backend::cpu {

namespace {

constexpr size_t kAlignment = 64;                  // matches GEMM PANEL_ALIGN_BYTES
constexpr size_t kMinClass  = 64;                  // smallest pooled size
constexpr size_t kMaxClass  = 16u * 1024u * 1024u; // 16 MiB cap
constexpr int    kNumClasses = 19;                 // 64 B .. 16 MiB (powers of two)

// Retention budget: total bytes allowed to sit in the central free lists. When
// a free pushes the pool over this bound, the largest pooled blocks are
// returned to the OS so the pool's footprint stays bounded. This is the knob
// that trades reuse (fewer malloc/free calls) against held-but-idle memory: a
// larger budget reuses more, a smaller budget returns scratch to the OS sooner.
constexpr size_t kMaxPooledBytes = 64u * 1024u * 1024u;

// Per-thread cache limit: the most payload a single thread may hold in its
// thread-local slots before further freed blocks are pushed to the central
// pool. One max-class block bounds worst-case retention across many threads.
constexpr size_t kMaxTlsBytes = 16u * 1024u * 1024u;

constexpr uint32_t kMagic = 0x4E4E4F50u;           // "NNOP" — block sanity tag

// Header stored immediately before the 64-byte-aligned payload. alignas(64)
// rounds sizeof up to 64, so `header + 1` lands the payload on a 64-byte
// boundary (the base is already 64-aligned from the aligned operator new).
struct alignas(kAlignment) PoolHeader {
    PoolHeader* next;    // central free-list link; nullptr while live / in a TLS slot
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

// A per-thread set of recycled blocks: one slot per size class. Operators
// allocate one scratch buffer per call and free it on return, so a thread that
// repeatedly hits the same size reuses its own slot with no lock and no
// central-pool traffic. Slots are atomics so release_all() can drain another
// thread's cache safely: exchange() gives each block to exactly one taker.
struct ThreadCache {
    std::array<std::atomic<PoolHeader*>, kNumClasses> free;
    std::atomic<size_t> bytes{0};    // payload bytes cached here (for stats)
    std::atomic<size_t> blocks{0};   // block count cached here (for stats)

    ThreadCache() {
        for (auto& slot : free) { slot.store(nullptr, std::memory_order_relaxed); }
    }
};

/// Points at this thread's cache once created; lives for the whole thread.
thread_local ThreadCache* t_cache = nullptr;

/// On thread exit, hand this thread's still-cached blocks back to the OS. The
/// cache node itself stays registered (owned by the pool) so other threads are
/// never left with a dangling pointer; it simply becomes empty.
struct TlsCacheReaper {
    ~TlsCacheReaper() {
        if (t_cache == nullptr) { return; }
        for (int i = 0; i < kNumClasses; ++i) {
            PoolHeader* h = t_cache->free[i].exchange(nullptr, std::memory_order_relaxed);
            if (h != nullptr) { ::operator delete(h, std::align_val_t{kAlignment}); }
        }
        t_cache->bytes.store(0, std::memory_order_relaxed);
        t_cache->blocks.store(0, std::memory_order_relaxed);
    }
};
thread_local TlsCacheReaper t_reaper;

}  // anonymous namespace

struct MemoryPool::Impl {
    std::array<PoolHeader*, kNumClasses> buckets{};  // central free lists

    std::mutex mutex;   // guards buckets + the central counters below
    size_t   central_pooled_bytes_  = 0;   // bytes in the central free lists
    size_t   central_pooled_blocks_ = 0;   // blocks in the central free lists

    // Shared counters touched by both the lock-free thread-local path and the
    // locked central path; relaxed atomics suffice (statistics only).
    std::atomic<size_t>   live_bytes_{0};
    std::atomic<uint64_t> allocations_{0};
    std::atomic<uint64_t> reuses_{0};

    // Registry of every thread's cache, so release_all() can drain them all.
    std::mutex registry_mutex;        // guards thread_caches_
    std::vector<ThreadCache*> thread_caches_;
};

MemoryPool::MemoryPool() : impl_(new Impl) {}

MemoryPool::~MemoryPool() {
    // Process exit: no concurrent operators remain; drain pooled memory back to
    // the OS. The ThreadCache registry nodes are intentionally not deleted: each
    // was already drained by its thread's reaper, and freeing them here could
    // race a reaper that runs after this destructor (thread-local vs. static
    // destruction order is not guaranteed). They are ~200 B apiece, bounded by
    // the peak thread count, and reclaimed by the OS at exit.
    release_all();
    delete impl_;
    impl_ = nullptr;
}

MemoryPool& MemoryPool::instance() {
    static MemoryPool pool;
    return pool;
}

void* MemoryPool::allocate(size_t size) {
    impl_->allocations_.fetch_add(1, std::memory_order_relaxed);

    // Requests above the cap bypass the pool entirely (direct aligned alloc,
    // never recycled). Assumed not to happen for scratch, but kept as a guard.
    if (size > kMaxClass) {
        PoolHeader* h = static_cast<PoolHeader*>(
            ::operator new(sizeof(PoolHeader) + size, std::align_val_t{kAlignment}));
        h->next = nullptr;
        h->size = size;
        h->pooled = 0;
        h->magic = kMagic;
        impl_->live_bytes_.fetch_add(size, std::memory_order_relaxed);
        return h + 1;
    }

    const size_t cls = round_up_class(size);
    const int idx = class_index(cls);

    // Lazily create + register this thread's cache on first use.
    ThreadCache* c = t_cache;
    if (c == nullptr) {
        c = new ThreadCache;
        t_cache = c;
        std::lock_guard<std::mutex> reg(impl_->registry_mutex);
        impl_->thread_caches_.push_back(c);
    }

    // Fast path: serve from this thread's cache, no lock.
    PoolHeader* h = c->free[idx].exchange(nullptr, std::memory_order_relaxed);
    if (h != nullptr) {
        c->bytes.fetch_sub(cls, std::memory_order_relaxed);
        c->blocks.fetch_sub(1, std::memory_order_relaxed);
        impl_->reuses_.fetch_add(1, std::memory_order_relaxed);
        impl_->live_bytes_.fetch_add(cls, std::memory_order_relaxed);
        h->next = nullptr;
        return h + 1;
    }

    // Slow path: the central size-class free list.
    std::lock_guard<std::mutex> lock(impl_->mutex);
    h = impl_->buckets[idx];
    if (h != nullptr) {
        impl_->buckets[idx] = h->next;
        impl_->central_pooled_bytes_ -= cls;
        --impl_->central_pooled_blocks_;
        impl_->reuses_.fetch_add(1, std::memory_order_relaxed);
        impl_->live_bytes_.fetch_add(cls, std::memory_order_relaxed);
        h->next = nullptr;
        return h + 1;
    }

    PoolHeader* fresh = static_cast<PoolHeader*>(
        ::operator new(sizeof(PoolHeader) + cls, std::align_val_t{kAlignment}));
    fresh->next = nullptr;
    fresh->size = cls;
    fresh->pooled = 1;
    fresh->magic = kMagic;
    impl_->live_bytes_.fetch_add(cls, std::memory_order_relaxed);
    return fresh + 1;
}

void MemoryPool::deallocate(void* ptr) noexcept {
    if (ptr == nullptr) { return; }
    PoolHeader* h = static_cast<PoolHeader*>(ptr) - 1;
    NNOPS_ASSERT(h->magic == kMagic);  // double-free / foreign pointer
    const size_t size = h->size;       // read before h is possibly recycled below

    if (!h->pooled) {
        ::operator delete(h, std::align_val_t{kAlignment});
    } else {
        const int idx = class_index(size);

        // Lazily create + register this thread's cache on first use.
        ThreadCache* c = t_cache;
        if (c == nullptr) {
            c = new ThreadCache;
            t_cache = c;
            std::lock_guard<std::mutex> reg(impl_->registry_mutex);
            impl_->thread_caches_.push_back(c);
        }

        h->next = nullptr;

        // Cache in this thread's slot iff it is empty AND the thread is under
        // its local budget; otherwise the block goes to the central free list
        // (and may be trimmed there). The CAS makes the "empty" check atomic so
        // a concurrent release_all() can never be overwritten by a lost block.
        PoolHeader* expected = nullptr;
        if (c->bytes.load(std::memory_order_relaxed) + size <= kMaxTlsBytes &&
            c->free[idx].compare_exchange_strong(expected, h,
                                                 std::memory_order_relaxed)) {
            c->bytes.fetch_add(size, std::memory_order_relaxed);
            c->blocks.fetch_add(1, std::memory_order_relaxed);
        } else {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            h->next = impl_->buckets[idx];
            impl_->buckets[idx] = h;
            impl_->central_pooled_bytes_ += size;
            ++impl_->central_pooled_blocks_;

            // Bound the central footprint: return the largest surplus blocks to
            // the OS, largest-first, so the small hot classes are kept intact.
            for (int i = kNumClasses - 1;
                 i >= 0 && impl_->central_pooled_bytes_ > kMaxPooledBytes; --i) {
                const size_t cls = size_t{kMinClass} << i;
                PoolHeader* cur = impl_->buckets[i];
                while (cur != nullptr &&
                       impl_->central_pooled_bytes_ > kMaxPooledBytes) {
                    PoolHeader* next = cur->next;
                    ::operator delete(cur, std::align_val_t{kAlignment});
                    --impl_->central_pooled_blocks_;
                    impl_->central_pooled_bytes_ -= cls;
                    cur = next;
                }
                impl_->buckets[i] = cur;
            }
        }
    }
    impl_->live_bytes_.fetch_sub(size, std::memory_order_relaxed);
}

void MemoryPool::release_all() {
    // Drain every thread's cache first, then the central free lists. Assumes no
    // concurrent allocate/deallocate while releasing (a force-release / teardown
    // operation), so the two locks are taken one at a time, never nested.
    {
        std::lock_guard<std::mutex> lock(impl_->registry_mutex);
        for (ThreadCache* c : impl_->thread_caches_) {
            for (int i = 0; i < kNumClasses; ++i) {
                PoolHeader* h = c->free[i].exchange(nullptr, std::memory_order_relaxed);
                if (h != nullptr) {
                    ::operator delete(h, std::align_val_t{kAlignment});
                }
            }
            c->bytes.store(0, std::memory_order_relaxed);
            c->blocks.store(0, std::memory_order_relaxed);
        }
    }

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
    impl_->central_pooled_bytes_ = 0;
    impl_->central_pooled_blocks_ = 0;
}

size_t MemoryPool::live_bytes() const noexcept {
    return impl_->live_bytes_.load(std::memory_order_relaxed);
}
size_t MemoryPool::pooled_bytes() const noexcept {
    size_t tls = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->registry_mutex);
        for (const ThreadCache* c : impl_->thread_caches_) {
            tls += c->bytes.load(std::memory_order_relaxed);
        }
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return tls + impl_->central_pooled_bytes_;
}
size_t MemoryPool::pooled_blocks() const noexcept {
    size_t tls = 0;
    {
        std::lock_guard<std::mutex> lock(impl_->registry_mutex);
        for (const ThreadCache* c : impl_->thread_caches_) {
            tls += c->blocks.load(std::memory_order_relaxed);
        }
    }
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return tls + impl_->central_pooled_blocks_;
}
uint64_t MemoryPool::allocation_count() const noexcept {
    return impl_->allocations_.load(std::memory_order_relaxed);
}
uint64_t MemoryPool::reuse_count() const noexcept {
    return impl_->reuses_.load(std::memory_order_relaxed);
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