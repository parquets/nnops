#pragma once
/// @file small_vector.hpp
/// @brief SmallVector — a vector with inline storage for up to N elements.
///
/// For tensors with rank ≤ N (N=8 by default), no heap allocation occurs.
/// This covers all practical neural network tensor dimensionalities.

#include "nnops/detail/assert.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <span>

namespace nnops::detail {

template <typename T, size_t N>
class SmallVector {
public:
    using value_type      = T;
    using size_type       = size_t;
    using difference_type = std::ptrdiff_t;
    using reference       = T&;
    using const_reference = const T&;
    using iterator        = T*;
    using const_iterator  = const T*;

    // ---- Construction ----
    constexpr SmallVector() = default;

    constexpr SmallVector(std::initializer_list<T> il) noexcept {
        NNOPS_ASSERT(il.size() <= N);
        auto* p = storage_.data();
        for (const auto& v : il) {
            *p++ = v;
        }
        size_ = il.size();
    }

    explicit constexpr SmallVector(std::span<const T> s) noexcept {
        NNOPS_ASSERT(s.size() <= N);
        auto* p = storage_.data();
        for (const auto& v : s) {
            *p++ = v;
        }
        size_ = s.size();
    }

    // ---- Capacity ----
    constexpr size_t size() const noexcept { return size_; }
    constexpr bool empty() const noexcept { return size_ == 0; }
    static constexpr size_t capacity() noexcept { return N; }
    static constexpr size_t max_size() noexcept { return N; }

    // ---- Element access ----
    constexpr T& operator[](size_t i) noexcept {
        NNOPS_ASSERT(i < size_);
        return storage_[i];
    }
    constexpr const T& operator[](size_t i) const noexcept {
        NNOPS_ASSERT(i < size_);
        return storage_[i];
    }
    constexpr T& front() noexcept { return storage_[0]; }
    constexpr const T& front() const noexcept { return storage_[0]; }
    constexpr T& back() noexcept { return storage_[size_ - 1]; }
    constexpr const T& back() const noexcept { return storage_[size_ - 1]; }
    constexpr T* data() noexcept { return storage_.data(); }
    constexpr const T* data() const noexcept { return storage_.data(); }

    // ---- Iterators ----
    constexpr iterator begin() noexcept { return storage_.data(); }
    constexpr iterator end() noexcept { return storage_.data() + size_; }
    constexpr const_iterator begin() const noexcept { return storage_.data(); }
    constexpr const_iterator end() const noexcept { return storage_.data() + size_; }
    constexpr const_iterator cbegin() const noexcept { return storage_.data(); }
    constexpr const_iterator cend() const noexcept { return storage_.data() + size_; }

    // ---- Modifiers ----
    constexpr void push_back(const T& val) noexcept {
        NNOPS_ASSERT(size_ < N);
        storage_[size_++] = val;
    }

    constexpr void pop_back() noexcept {
        NNOPS_ASSERT(size_ > 0);
        --size_;
    }

    constexpr void clear() noexcept { size_ = 0; }

    constexpr void resize(size_t count) noexcept {
        NNOPS_ASSERT(count <= N);
        size_ = count;
    }

    // Assignment from span
    constexpr SmallVector& operator=(std::span<const T> s) noexcept {
        NNOPS_ASSERT(s.size() <= N);
        auto* p = storage_.data();
        for (const auto& v : s) {
            *p++ = v;
        }
        size_ = s.size();
        return *this;
    }

    // ---- Comparison ----
    constexpr bool operator==(const SmallVector& other) const noexcept {
        if (size_ != other.size_) { return false; }
        for (size_t i = 0; i < size_; ++i) {
            if (storage_[i] != other.storage_[i]) { return false; }
        }
        return true;
    }

    constexpr bool operator!=(const SmallVector& other) const noexcept {
        return !(*this == other);
    }

private:
    std::array<T, N> storage_{};
    size_t size_ = 0;
};

}  // namespace nnops::detail
