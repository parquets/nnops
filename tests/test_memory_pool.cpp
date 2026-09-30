/// @file test_memory_pool.cpp
/// @brief Tests for the CPU backend scratch-memory pool (MemoryPool singleton).

#include "backend/cpu/common/memory_pool.hpp"
#include "nnops/core/backend.hpp"
#include "common/test_harness.hpp"

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

using nnops::backend::cpu::MemoryPool;

namespace {

constexpr size_t kAlign = 64;
constexpr size_t kOversize = 20u * 1024u * 1024u;  // > 16 MiB cap

inline bool aligned(void* p) {
    return (reinterpret_cast<uintptr_t>(p) % kAlign) == 0;
}

}  // anonymous namespace

NNOPS_TEST(memory_pool_alignment_and_sizes) {
    auto& pool = MemoryPool::instance();
    pool.release_all();

    // Smallest and a few representative sizes must all be 64-byte aligned.
    for (size_t size : {size_t{1}, size_t{64}, size_t{100}, size_t{1024},
                        size_t{65536}, size_t{4u * 1024u * 1024u}}) {
        void* p = pool.allocate(size);
        NNOPS_EXPECT_TRUE(aligned(p));
        pool.deallocate(p);
    }

    // A block written and read back holds its bytes.
    int* arr = pool.allocate_typed<int>(100);
    for (int i = 0; i < 100; ++i) { arr[i] = i; }
    for (int i = 0; i < 100; ++i) { NNOPS_EXPECT_EQ(arr[i], i); }
    pool.deallocate(arr);

    NNOPS_EXPECT_EQ(pool.live_bytes(), size_t{0});
}

NNOPS_TEST(memory_pool_reuses_freed_blocks) {
    auto& pool = MemoryPool::instance();
    pool.release_all();

    void* p1 = pool.allocate(4096);
    pool.deallocate(p1);

    // Same size class: the next allocation should recycle the same address.
    void* p2 = pool.allocate(4096);
    NNOPS_EXPECT_EQ(p1, p2);

    NNOPS_EXPECT_EQ(pool.pooled_blocks(), size_t{0});   // p2 is now live

    // Rounding to the size class: 100 B and 65 B both land in the 128 B class,
    // so freeing 100 B can satisfy a 65 B request.
    void* q1 = pool.allocate(100);
    pool.deallocate(q1);
    void* q2 = pool.allocate(65);
    NNOPS_EXPECT_EQ(q1, q2);

    pool.deallocate(p2);
    pool.deallocate(q2);
    NNOPS_EXPECT_EQ(pool.live_bytes(), size_t{0});
}

NNOPS_TEST(memory_pool_oversize_not_pooled) {
    auto& pool = MemoryPool::instance();
    pool.release_all();

    void* p = pool.allocate(kOversize);
    NNOPS_EXPECT_TRUE(aligned(p));
    const size_t pooled_before = pool.pooled_blocks();

    pool.deallocate(p);
    // Oversize blocks go straight back to the OS, never into a free list.
    NNOPS_EXPECT_EQ(pool.pooled_blocks(), pooled_before);
    NNOPS_EXPECT_EQ(pool.live_bytes(), size_t{0});
}

NNOPS_TEST(memory_pool_concurrent_alloc_free) {
    auto& pool = MemoryPool::instance();
    pool.release_all();

    const int nthreads = 8;
    const int iters = 2000;
    std::atomic<bool> ok{true};

    std::vector<std::thread> workers;
    workers.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        workers.emplace_back([&pool, &ok, t]() {
            // Stagger sizes so different threads hit different size classes.
            const size_t base = 64u << (t % 8);
            for (int i = 0; i < iters; ++i) {
                const size_t size = base + (i % 64);
                void* p = pool.allocate(size);
                if (!aligned(p)) { ok.store(false, std::memory_order_relaxed); }
                // Touch the memory so the OS actually maps it.
                static_cast<uint8_t*>(p)[0] = static_cast<uint8_t>(i);
                static_cast<uint8_t*>(p)[size - 1] = static_cast<uint8_t>(t);
                pool.deallocate(p);
            }
        });
    }
    for (auto& w : workers) { w.join(); }

    NNOPS_EXPECT_TRUE(ok.load());
    // Every block was freed; nothing should remain handed out.
    NNOPS_EXPECT_EQ(pool.live_bytes(), size_t{0});
}

NNOPS_TEST(memory_pool_poolptr_raii) {
    auto& pool = MemoryPool::instance();
    pool.release_all();
    const size_t live_before = pool.live_bytes();

    {
        nnops::backend::cpu::PoolPtr a(1024);
        NNOPS_EXPECT_TRUE(aligned(a.get()));
        NNOPS_EXPECT_TRUE(static_cast<bool>(a));
        NNOPS_EXPECT_EQ(pool.live_bytes(), live_before + 1024);

        // Move semantics: a is emptied, b takes over the block.
        nnops::backend::cpu::PoolPtr b(std::move(a));
        NNOPS_EXPECT_FALSE(static_cast<bool>(a));
        NNOPS_EXPECT_TRUE(static_cast<bool>(b));
    }  // both scopes exit → block freed

    NNOPS_EXPECT_EQ(pool.live_bytes(), live_before);
}

NNOPS_TEST(memory_pool_release_all) {
    auto& pool = MemoryPool::instance();
    pool.release_all();

    void* p = pool.allocate(8192);
    pool.deallocate(p);
    NNOPS_EXPECT_TRUE(pool.pooled_bytes() >= 8192);

    pool.release_all();
    NNOPS_EXPECT_EQ(pool.pooled_bytes(), size_t{0});
    NNOPS_EXPECT_EQ(pool.pooled_blocks(), size_t{0});

    // Still usable after release: a fresh allocation must be aligned and valid.
    void* q = pool.allocate(8192);
    NNOPS_EXPECT_TRUE(aligned(q));
    pool.deallocate(q);
    NNOPS_EXPECT_EQ(pool.live_bytes(), size_t{0});
}

NNOPS_TEST(memory_pool_public_release_scratch) {
    auto& pool = MemoryPool::instance();
    pool.release_all();

    void* p = pool.allocate(4096);
    pool.deallocate(p);
    NNOPS_EXPECT_TRUE(pool.pooled_bytes() >= 4096);

    // The public interface force-releases the same underlying pool.
    nnops::release_scratch_memory();
    NNOPS_EXPECT_EQ(pool.pooled_bytes(), size_t{0});
    NNOPS_EXPECT_EQ(pool.live_bytes(), size_t{0});
}
