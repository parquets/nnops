#pragma once
/// @file random_tensor.hpp
/// @brief Random data generation utilities for testing.
///
/// Uses XorShift128+ for fast, deterministic pseudo-random number generation.
/// No external dependencies.

#include "nnops/core/tensor_view.hpp"

#include <cstdint>
#include <vector>
#include <span>
#include <initializer_list>
#include <utility>

namespace nnops::test {

/// Fast, deterministic PRNG based on XorShift128+.
class XorShift128 {
public:
    explicit XorShift128(uint64_t seed = 42) noexcept {
        state_[0] = splitmix64(seed);
        state_[1] = splitmix64(state_[0]);
    }

    /// Generate next uint64_t.
    uint64_t next_u64() noexcept {
        uint64_t s1 = state_[0];
        const uint64_t s0 = state_[1];
        state_[0] = s0;
        s1 ^= s1 << 23;
        state_[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
        return state_[1] + s0;
    }

    /// Generate float in [min, max).
    float next_float(float min = -1.0f, float max = 1.0f) noexcept {
        // Generate in [0, 1) then scale
        const uint64_t bits = next_u64();
        const float t = static_cast<float>(bits & 0xFFFFFF) / 16777216.0f;
        return min + t * (max - min);
    }

    /// Fill a buffer with random float values.
    void fill_float(float* data, int64_t numel,
                    float min = -1.0f, float max = 1.0f) noexcept {
        for (int64_t i = 0; i < numel; ++i) {
            data[i] = next_float(min, max);
        }
    }

private:
    static uint64_t splitmix64(uint64_t x) noexcept {
        uint64_t z = (x + 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }

    uint64_t state_[2];
};

/// Create a random float tensor.
/// Returns a (vector, TensorView) pair. The vector owns the data.
inline std::pair<std::vector<float>, TensorView>
make_random_tensor(std::span<const int64_t> shape,
                   float min = -1.0f, float max = 1.0f,
                   uint64_t seed = 12345)
{
    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;

    std::vector<float> data(static_cast<size_t>(numel));
    XorShift128 rng(seed);
    rng.fill_float(data.data(), numel, min, max);

    TensorView view(shape, DataType::f32, data.data());
    return {std::move(data), view};
}

/// Overload for initializer_list convenience: make_random_tensor({1, 3, 32, 32}).
inline std::pair<std::vector<float>, TensorView>
make_random_tensor(std::initializer_list<int64_t> shape,
                   float min = -1.0f, float max = 1.0f,
                   uint64_t seed = 12345)
{
    return make_random_tensor(std::span<const int64_t>(shape.begin(), shape.size()),
                              min, max, seed);
}

/// Fill a pre-allocated TensorView with random data.
inline void fill_random_float(TensorView& tensor,
                               float min = -1.0f, float max = 1.0f,
                               uint64_t seed = 12345)
{
    XorShift128 rng(seed);
    rng.fill_float(tensor.data_as<float>(), tensor.numel(), min, max);
}

}  // namespace nnops::test
