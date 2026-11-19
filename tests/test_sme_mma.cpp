/// @file test_sme_mma.cpp
/// @brief The SME MMA micro-kernels against a scalar reference.
///
/// The kernels take the packed layouts directly, so the test packs with the
/// same packers the engine uses and checks every path: overwrite / accumulate /
/// bias-seeded accumulators, clamped stores, both dtypes, and non-multiple tile
/// tails. On a machine without SME the test reports and passes without calling
/// in.

#include "backend/cpu/aarch64/sme/mma_f16_sme.hpp"
#include "backend/cpu/aarch64/sme/mma_f32_sme.hpp"
#include "backend/cpu/aarch64/sme/mma_i8_sme.hpp"
#include "backend/cpu/aarch64/sme/mma_sme.hpp"
#include "backend/cpu/aarch64/sme/pack_f16_sme.hpp"
#include "backend/cpu/aarch64/sme/pack_f32_sme.hpp"
#include "backend/cpu/aarch64/sme/pack_i8_sme.hpp"

#include "common/test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#if defined(NNOPS_ARCH_SME)

using namespace nnops;
using nnops::backend::cpu::half;
using namespace nnops::backend::cpu::aarch64;
using nnops::backend::cpu::aarch64::available_sme;

namespace {

constexpr int kMr = 16;

/// A[m][k] = row-major mc x kc.
std::vector<float> make_a(int mc, int kc, uint32_t seed)
{
    std::vector<float> a(static_cast<size_t>(mc) * kc);
    uint32_t st = seed;
    for (auto& v : a) {
        st = st * 1664525u + 1013904223u;
        v = (static_cast<float>(st >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
    return a;
}

/// B[k][n] = row-major kc x nc.
std::vector<float> make_b(int nc, int kc, uint32_t seed)
{
    std::vector<float> b(static_cast<size_t>(nc) * kc);
    uint32_t st = seed;
    for (auto& v : b) {
        st = st * 1664525u + 1013904223u;
        v = (static_cast<float>(st >> 8) / 16777216.0f - 0.5f) * 2.0f;
    }
    return b;
}

float ref_at(const std::vector<float>& a, const std::vector<float>& b, int kc, int nc, int m,
             int n)
{
    float s = 0.0f;
    for (int k = 0; k < kc; ++k) {
        s += a[static_cast<size_t>(m) * kc + k] * b[static_cast<size_t>(k) * nc + n];
    }
    return s;
}

/// One shape through the f32 kernel, all three accumulator modes.
void run_f32(int mc, int nc, int kc)
{
    const auto a = make_a(mc, kc, static_cast<uint32_t>(mc * 7 + mc + kc));
    const auto b = make_b(nc, kc, static_cast<uint32_t>(nc * 3 + kc));
    std::vector<float> pa(packed_a_elems_f32_sme(mc, kc));
    std::vector<float> pb(packed_b_elems_f32_sme(nc, kc));
    pack_a_f32_sme(mc, kc, pa.data(), a.data(), kc);
    pack_b_f32_sme(nc, kc, pb.data(), b.data(), nc);

    std::vector<float> c(static_cast<size_t>(mc) * nc);

    // Overwrite, no clamp.
    std::fill(c.begin(), c.end(), 123.0f);
    mma_f32_sme(mc, nc, kc, c.data(), nc, pa.data(), pb.data(),
                                /*zero_mode=*/true, -1e30f, 1e30f, nullptr);
    for (int m = 0; m < mc; ++m) {
        for (int n = 0; n < nc; ++n) {
            const float want = ref_at(a, b, kc, nc, m, n);
            NNOPS_EXPECT_NEAR(c[static_cast<size_t>(m) * nc + n], want, 1e-4f);
        }
    }

    // Accumulate onto the caller's C.
    std::vector<float> c2(c.size());
    uint32_t st = 99u;
    for (auto& v : c2) {
        st = st * 1664525u + 1013904223u;
        v = static_cast<float>(st >> 8) / 16777216.0f;
    }
    std::vector<float> base = c2;
    mma_f32_sme(mc, nc, kc, c2.data(), nc, pa.data(), pb.data(),
                                /*zero_mode=*/false, -1e30f, 1e30f, nullptr);
    for (int m = 0; m < mc; ++m) {
        for (int n = 0; n < nc; ++n) {
            const float want = base[static_cast<size_t>(m) * nc + n] + ref_at(a, b, kc, nc, m, n);
            NNOPS_EXPECT_NEAR(c2[static_cast<size_t>(m) * nc + n], want, 1e-4f);
        }
    }

    // Bias-seeded accumulator, clamped store.
    std::vector<float> bias(static_cast<size_t>(nc));
    for (int n = 0; n < nc; ++n) { bias[static_cast<size_t>(n)] = 0.01f * static_cast<float>(n); }
    std::fill(c.begin(), c.end(), -777.0f);
    mma_f32_sme(mc, nc, kc, c.data(), nc, pa.data(), pb.data(),
                                /*zero_mode=*/true, -0.5f, 0.5f, bias.data());
    for (int m = 0; m < mc; ++m) {
        for (int n = 0; n < nc; ++n) {
            const float want =
                std::clamp(bias[static_cast<size_t>(n)] + ref_at(a, b, kc, nc, m, n), -0.5f, 0.5f);
            NNOPS_EXPECT_NEAR(c[static_cast<size_t>(m) * nc + n], want, 1e-4f);
        }
    }
}

/// f16 storage, f32 compute: same shapes, reference built from the same
/// rounded inputs, so only the summation order differs.
void run_f16(int mc, int nc, int kc)
{
    const auto a32 = make_a(mc, kc, static_cast<uint32_t>(mc * 11 + kc));
    const auto b32 = make_b(nc, kc, static_cast<uint32_t>(nc * 5 + kc));
    std::vector<half> a(a32.size()), b(b32.size());
    for (size_t i = 0; i < a32.size(); ++i) { a[i] = static_cast<half>(a32[i]); }
    for (size_t i = 0; i < b32.size(); ++i) { b[i] = static_cast<half>(b32[i]); }
    std::vector<float> af(a.size()), bf(b.size());
    for (size_t i = 0; i < a.size(); ++i) { af[i] = static_cast<float>(a[i]); }
    for (size_t i = 0; i < b.size(); ++i) { bf[i] = static_cast<float>(b[i]); }

    std::vector<half> pa(packed_a_elems_f16_sme(mc, kc));
    std::vector<half> pb(packed_b_elems_f16_sme(nc, kc));
    pack_a_f16_sme(mc, kc, pa.data(), a.data(), kc);
    pack_b_f16_sme(nc, kc, pb.data(), b.data(), nc);
    std::vector<half> c(static_cast<size_t>(mc) * nc);
    mma_f16_sme(mc, nc, kc, c.data(), nc, pa.data(), pb.data(),
                                /*zero_mode=*/true, -1e30f, 1e30f, nullptr);
    for (int m = 0; m < mc; ++m) {
        for (int n = 0; n < nc; ++n) {
            const float want = ref_at(af, bf, kc, nc, m, n);
            // C is f16: the comparison sees the narrowed result, so the bound
            // is half an f16 ulp at the value's magnitude.
            NNOPS_EXPECT_NEAR(static_cast<float>(c[static_cast<size_t>(m) * nc + n]), want,
                              2e-3f);
        }
    }
}

/// int8: 4-way dot (1024 MACs per instruction), so tails on every axis and a K
/// that is not a multiple of four are the interesting cases.
void run_i8(int mc, int nc, int kc)
{
    std::vector<int8_t> a(static_cast<size_t>(mc) * kc), b(static_cast<size_t>(kc) * nc);
    uint32_t st = 1234u;
    for (auto& v : a) { st = st * 1664525u + 1013904223u; v = static_cast<int8_t>((st >> 16) % 251 - 125); }
    for (auto& v : b) { st = st * 1664525u + 1013904223u; v = static_cast<int8_t>((st >> 16) % 251 - 125); }

    std::vector<int8_t> pa(packed_a_elems_i8_sme(mc, kc));
    std::vector<int8_t> pb(packed_b_elems_i8_sme(nc, kc));
    pack_a_i8_sme(mc, kc, pa.data(), a.data(), kc, false, false);
    pack_b_i8_sme(nc, kc, pb.data(), b.data(), nc, false);

    std::vector<int32_t> c(static_cast<size_t>(mc) * nc);
    mma_i8_sme(mc, nc, kc, c.data(), nc, pa.data(), pb.data(),
                               /*zero_mode=*/true, INT32_MIN, INT32_MAX);
    for (int m = 0; m < mc; ++m) {
        for (int n = 0; n < nc; ++n) {
            int32_t want = 0;
            for (int k = 0; k < kc; ++k) {
                want += static_cast<int32_t>(a[static_cast<size_t>(m) * kc + k]) *
                        static_cast<int32_t>(b[static_cast<size_t>(k) * nc + n]);
            }
            NNOPS_EXPECT_EQ(c[static_cast<size_t>(m) * nc + n], want);
        }
    }
}

} // namespace

NNOPS_TEST(sme_mma_matches_reference) {
    if (!available_sme()) {
        std::printf("  (no SME on this build or CPU; skipped)\n");
        return;
    }
    // Full blocks, tails on every axis, and a K pair with no partner.
    run_f32(64, 64, 32);
    run_f32(40, 70, 17); // m and n tails, odd kc
    run_f32(16, 12, 1);  // single panel, k = 1
    run_f32(1, 13, 5);   // one row
    run_i8(64, 64, 32);   // full tiles
    run_i8(40, 70, 17);   // tails on every axis, K not a multiple of four
    run_i8(16, 12, 1);    // one K group, one panel
    run_i8(1, 13, 5);     // one row
    run_f16(64, 64, 32);
    run_f16(40, 70, 17); // odd kc pads the last K pair
    run_f16(1, 13, 5);
}

#endif // NNOPS_ARCH_SME
