/// @file test_mma_packed.cpp
/// @brief Correctness tests for the x86_64 / aarch64 packed-AB and packed-B
///        mrxncxkc MMA kernels.
///
/// Covers f32 and f16 at the arch's MR levels (x86_64: {6, 4, 2, 1}; aarch64:
/// {8, 4, 1}) against a uniform packed-B panel width (x86_64: 16; aarch64: 12
/// for f32, 16 for f16). Every kernel takes a runtime `mr`, so a tile of any
/// height up to its own MR is served by the same code: rows past `mr` alias row
/// mr-1 in A and C, compute a duplicate of it, and store it back to the same
/// place. B is laid out as the packer writes it: `ceil(nc/panel)` panel-wide
/// slots, the last one zero-padded past `nc`, each in one aligned slot.
///
/// Each case checks:
///   - live rows/columns match a scalar reference (clamp, beta == 1 add, and
///     for f16 the round-trip through fp16 storage),
///   - columns >= nc are untouched,
///   - rows >= mr are untouched (they share row mr-1's C pointer),
///   - the C tail and the B tail keep their sentinels.

#include "common/test_harness.hpp"
#include "nnops/detail/simd.hpp"
#include "nnops/detail/half.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

#if defined(NNOPS_ARCH_X86_64) || defined(NNOPS_ARCH_AARCH64)

  #if defined(NNOPS_ARCH_X86_64)
    #include "backend/cpu/x86_64/mma_f32.hpp"
    #include "backend/cpu/x86_64/mma_f16.hpp"
  #else
    #include "backend/cpu/aarch64/mma_f32.hpp"
    #include "backend/cpu/aarch64/mma_f16.hpp"
  #endif

namespace {

#if defined(NNOPS_ARCH_X86_64)
using namespace nnops::backend::cpu::x86_64;
#else
using namespace nnops::backend::cpu::aarch64;
#endif
using nnops::backend::cpu::half;
using nnops::backend::cpu::half_to_float;
using nnops::backend::cpu::float_to_half;
using nnops::backend::cpu::mma_pack_align;

constexpr float C_SENTINEL = -777.f;
constexpr float B_SENTINEL = 12345.f;

// ---- dtype bridges --------------------------------------------------------

template <typename T> float to_float(T v) noexcept;
template <> float to_float<float>(float v) noexcept { return v; }
template <> float to_float<half>(half v) noexcept { return half_to_float(v); }

template <typename T> T from_float(float v) noexcept;
template <> float from_float<float>(float v) noexcept { return v; }
template <> half from_float<half>(float v) noexcept { return float_to_half(v); }

template <typename T> float tolerance() noexcept;
template <> float tolerance<float>() noexcept { return 1e-3f; }
#if defined(NNOPS_ARCH_X86_64)
template <> float tolerance<half>() noexcept { return 2e-3f; }   // f32 accumulate
#else
template <> float tolerance<half>() noexcept { return 2e-2f; }   // f16 accumulate
#endif

// ---- arch tile shape ------------------------------------------------------

#if defined(NNOPS_ARCH_X86_64)
constexpr int MR_LEVELS[4] = {6, 4, 2, 1};
constexpr int MR_COUNT = 4;
#else
constexpr int MR_LEVELS[3] = {8, 4, 1};
constexpr int MR_COUNT = 3;
#endif

/// Uniform packed-B panel width: the B lanes the kernels read per k.
template <typename T> constexpr int panel_width() noexcept {
    if constexpr (std::is_same_v<T, float>) {
#if defined(NNOPS_ARCH_X86_64)
        return 16;
#else
        return 12;
#endif
    } else {
        return 16;
    }
}

/// Kernel signatures at every tile height (mr is first, as in XNNPACK).
template <typename T>
using AbKernel = void (*)(int, int, int, T*, int, const T*, const T*, float, float);
template <typename T>
using BKernel  = void (*)(int, int, int, T*, int, const T*, int, const T*,
                          float, float);

// ---- level tables ---------------------------------------------------------

enum class Kind { Ab, B };

template <typename T>
auto ab_fn(int level, bool zero_mode) {
#if defined(NNOPS_ARCH_X86_64)
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 6: return zero_mode ? mma_packed_ab_mr6_nr16_mrxncxkc_f32<true>
                                 : mma_packed_ab_mr6_nr16_mrxncxkc_f32<false>;
        case 4: return zero_mode ? mma_packed_ab_mr4_nr16_mrxncxkc_f32<true>
                                 : mma_packed_ab_mr4_nr16_mrxncxkc_f32<false>;
        case 2: return zero_mode ? mma_packed_ab_mr2_nr16_mrxncxkc_f32<true>
                                 : mma_packed_ab_mr2_nr16_mrxncxkc_f32<false>;
        default: return zero_mode ? mma_packed_ab_mr1_nr16_mrxncxkc_f32<true>
                                  : mma_packed_ab_mr1_nr16_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 6: return zero_mode ? mma_packed_ab_mr6_nr16_mrxncxkc_f16<true>
                                 : mma_packed_ab_mr6_nr16_mrxncxkc_f16<false>;
        case 4: return zero_mode ? mma_packed_ab_mr4_nr16_mrxncxkc_f16<true>
                                 : mma_packed_ab_mr4_nr16_mrxncxkc_f16<false>;
        case 2: return zero_mode ? mma_packed_ab_mr2_nr16_mrxncxkc_f16<true>
                                 : mma_packed_ab_mr2_nr16_mrxncxkc_f16<false>;
        default: return zero_mode ? mma_packed_ab_mr1_nr16_mrxncxkc_f16<true>
                                  : mma_packed_ab_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#else
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 8: return zero_mode ? mma_packed_ab_mr8_nr12_mrxncxkc_f32<true>
                                 : mma_packed_ab_mr8_nr12_mrxncxkc_f32<false>;
        case 4: return zero_mode ? mma_packed_ab_mr4_nr12_mrxncxkc_f32<true>
                                 : mma_packed_ab_mr4_nr12_mrxncxkc_f32<false>;
        default: return zero_mode ? mma_packed_ab_mr1_nr12_mrxncxkc_f32<true>
                                  : mma_packed_ab_mr1_nr12_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 8: return zero_mode ? mma_packed_ab_mr8_nr16_mrxncxkc_f16<true>
                                 : mma_packed_ab_mr8_nr16_mrxncxkc_f16<false>;
        case 4: return zero_mode ? mma_packed_ab_mr4_nr16_mrxncxkc_f16<true>
                                 : mma_packed_ab_mr4_nr16_mrxncxkc_f16<false>;
        default: return zero_mode ? mma_packed_ab_mr1_nr16_mrxncxkc_f16<true>
                                  : mma_packed_ab_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#endif
}

template <typename T>
auto b_fn(int level, bool zero_mode) {
#if defined(NNOPS_ARCH_X86_64)
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 6: return zero_mode ? mma_packed_b_mr6_nr16_mrxncxkc_f32<true>
                                 : mma_packed_b_mr6_nr16_mrxncxkc_f32<false>;
        case 4: return zero_mode ? mma_packed_b_mr4_nr16_mrxncxkc_f32<true>
                                 : mma_packed_b_mr4_nr16_mrxncxkc_f32<false>;
        case 2: return zero_mode ? mma_packed_b_mr2_nr16_mrxncxkc_f32<true>
                                 : mma_packed_b_mr2_nr16_mrxncxkc_f32<false>;
        default: return zero_mode ? mma_packed_b_mr1_nr16_mrxncxkc_f32<true>
                                  : mma_packed_b_mr1_nr16_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 6: return zero_mode ? mma_packed_b_mr6_nr16_mrxncxkc_f16<true>
                                 : mma_packed_b_mr6_nr16_mrxncxkc_f16<false>;
        case 4: return zero_mode ? mma_packed_b_mr4_nr16_mrxncxkc_f16<true>
                                 : mma_packed_b_mr4_nr16_mrxncxkc_f16<false>;
        case 2: return zero_mode ? mma_packed_b_mr2_nr16_mrxncxkc_f16<true>
                                 : mma_packed_b_mr2_nr16_mrxncxkc_f16<false>;
        default: return zero_mode ? mma_packed_b_mr1_nr16_mrxncxkc_f16<true>
                                  : mma_packed_b_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#else
    if constexpr (std::is_same_v<T, float>) {
        switch (level) {
        case 8: return zero_mode ? mma_packed_b_mr8_nr12_mrxncxkc_f32<true>
                                 : mma_packed_b_mr8_nr12_mrxncxkc_f32<false>;
        case 4: return zero_mode ? mma_packed_b_mr4_nr12_mrxncxkc_f32<true>
                                 : mma_packed_b_mr4_nr12_mrxncxkc_f32<false>;
        default: return zero_mode ? mma_packed_b_mr1_nr12_mrxncxkc_f32<true>
                                  : mma_packed_b_mr1_nr12_mrxncxkc_f32<false>;
        }
    } else {
        switch (level) {
        case 8: return zero_mode ? mma_packed_b_mr8_nr16_mrxncxkc_f16<true>
                                 : mma_packed_b_mr8_nr16_mrxncxkc_f16<false>;
        case 4: return zero_mode ? mma_packed_b_mr4_nr16_mrxncxkc_f16<true>
                                 : mma_packed_b_mr4_nr16_mrxncxkc_f16<false>;
        default: return zero_mode ? mma_packed_b_mr1_nr16_mrxncxkc_f16<true>
                                  : mma_packed_b_mr1_nr16_mrxncxkc_f16<false>;
        }
    }
#endif
}

// ---- case construction ----------------------------------------------------

float rnd() { return static_cast<float>(std::rand() % 2001 - 1000) / 250.f; }

/// Number of uniform packed-B panels the packer lays down for `nc` columns.
int panel_slots(int nc, int panel) { return (nc + panel - 1) / panel; }

/// B in packed-panel form: every slot is a full `panel`-wide panel in an `ldb`
/// slot, with the columns past `nc` zero-filled, and a sentinel tail after the
/// last slot. The uniform width is what lets the kernel read `panel` values per
/// k even for a narrow tail.
template <typename T>
std::vector<T> make_packed_b(const std::vector<T>& b_raw, int nc, int kc,
                             int ldb, int nslots, int panel) {
    std::vector<T> b(static_cast<size_t>(nslots) * ldb + 64, from_float<T>(B_SENTINEL));
    const T zero = from_float<T>(0.f);
    for (int p = 0; p < nslots; ++p) {
        const int col0 = panel * p;
        const int w = std::min(panel, nc - col0);
        T* slot = b.data() + static_cast<size_t>(p) * ldb;
        for (int k = 0; k < kc; ++k) {
            for (int j = 0; j < w; ++j) {
                slot[k * panel + j] = b_raw[static_cast<size_t>(k) * nc + col0 + j];
            }
            for (int j = w; j < panel; ++j) { slot[k * panel + j] = zero; }
        }
    }
    return b;
}

/// Shared result check. `a_at(k, r)` returns the A element the reference
/// multiplies, which is [kc][MR] packed for an AB kernel and [r][k] row-major
/// for a B kernel.
template <typename T, typename AAt>
void verify(const std::vector<T>& c, const std::vector<T>& c0,
            const std::vector<T>& b, const std::vector<T>& b_raw,
            const std::vector<T>& a, int nc, int kc, int ldc, int mr, int MR,
            int ldb, int nslots, float clamp_min, float clamp_max,
            bool zero_mode, Kind kind, AAt a_at, const char* what) {
    const float tol = tolerance<T>();
    const int live = std::min(mr, MR);
    // Sentinels as they actually sit in T: f16 may round the literals.
    const float c_fill = to_float(from_float<T>(C_SENTINEL));
    const float b_fill = to_float(from_float<T>(B_SENTINEL));

    auto fail = [&](const char* why, int r, int col) {
        char msg[320];
        std::snprintf(msg, sizeof(msg),
                      "%s: %s at mr=%d/%d nc=%d kc=%d ldc=%d r=%d c=%d",
                      what, why, mr, MR, nc, kc, ldc, r, col);
        throw std::runtime_error(msg);
    };
    auto fail_value = [&](int r, int col, float got, float ref) {
        char msg[320];
        std::snprintf(msg, sizeof(msg),
                      "%s: value at mr=%d/%d nc=%d kc=%d ldc=%d r=%d c=%d "
                      "(got %.6f want %.6f)",
                      what, mr, MR, nc, kc, ldc, r, col, got, ref);
        throw std::runtime_error(msg);
    };

    // aarch64 fp16 kernels accumulate in f16 (x86_64 and every f32 kernel
    // accumulate in f32). Match that in the reference: kc products of mixed
    // sign can cancel to near zero, and an f32 reference then disagrees with
    // the f16 kernel by far more than the relative tolerance allows.
#if defined(NNOPS_ARCH_AARCH64)
    constexpr bool f16_accum = std::is_same_v<T, half>;
#else
    constexpr bool f16_accum = false;
#endif

    for (int r = 0; r < live; ++r) {
        for (int col = 0; col < nc; ++col) {
            float acc;
            if constexpr (f16_accum) {
                T h = from_float<T>(0.f);
                for (int k = 0; k < kc; ++k) {
                    h = from_float<T>(
                        to_float(h) +
                        to_float(a_at(k, r)) *
                            to_float(b_raw[static_cast<size_t>(k) * nc + col]));
                }
                acc = to_float(h);
            } else {
                acc = 0.f;
                for (int k = 0; k < kc; ++k) {
                    acc += to_float(a_at(k, r)) *
                           to_float(b_raw[static_cast<size_t>(k) * nc + col]);
                }
            }
            float ref;
            if constexpr (f16_accum) {
                const float acc_c = zero_mode
                    ? acc
                    : to_float(from_float<T>(
                          to_float(c0[static_cast<size_t>(r) * ldc + col]) + acc));
                ref = acc_c;
            } else {
                ref = zero_mode
                    ? acc
                    : to_float(c0[static_cast<size_t>(r) * ldc + col]) + acc;
            }
            ref = std::fmin(std::fmax(ref, clamp_min), clamp_max);
            ref = to_float(from_float<T>(ref));          // f16 storage rounding
            const float got = to_float(c[static_cast<size_t>(r) * ldc + col]);
            if (!(std::fabs(got - ref) <= tol * (1.f + std::fabs(ref)))) {
                fail_value(r, col, got, ref);
            }
        }
        for (int col = nc; col < ldc; ++col) {           // untouched columns
            if (to_float(c[static_cast<size_t>(r) * ldc + col]) != c_fill) {
                fail("column past nc", r, col);
            }
        }
    }

    for (int r = live; r < MR; ++r) {                    // aliased rows never store
        for (int col = 0; col < ldc; ++col) {
            const size_t i = static_cast<size_t>(r) * ldc + col;
            if (to_float(c[i]) != to_float(c0[i])) { fail("row past mr", r, col); }
        }
    }

    for (size_t i = static_cast<size_t>(ldc) * MR; i < c.size(); ++i) {
        if (to_float(c[i]) != c_fill) { fail("C tail", 0, 0); }
    }

    for (size_t i = static_cast<size_t>(nslots) * ldb; i < b.size(); ++i) {
        if (to_float(b[i]) != b_fill) { fail("B tail", 0, 0); }
    }
    (void)a;
    (void)kind;
}

/// One packed-AB case: A is [kc][MR], B is packed.
template <typename T>
void check_ab(AbKernel<T> fn, int level, bool zero_mode, int mr, int nc, int kc,
              int ldc, float clamp_min, float clamp_max) {
    const int panel = panel_width<T>();
    const int ldb = static_cast<int>(mma_pack_align(panel * kc * sizeof(T)) / sizeof(T));
    const int nslots = panel_slots(nc, panel);

    std::vector<T> a(static_cast<size_t>(kc) * level);
    for (auto& v : a) { v = from_float<T>(rnd()); }
    // Rows past `mr` alias row mr-1, exactly as pack_trans_* lays them down:
    // the AB kernels read all MR lanes of A, and the extra rows must compute
    // duplicates that the aliased C row pointers can store idempotently.
    const int last = (mr > 1) ? mr - 1 : 0;
    for (int k = 0; k < kc; ++k) {
        for (int r = mr; r < level; ++r) {
            a[static_cast<size_t>(k) * level + r] =
                a[static_cast<size_t>(k) * level + last];
        }
    }

    std::vector<T> b_raw(static_cast<size_t>(kc) * nc);
    for (auto& v : b_raw) { v = from_float<T>(rnd()); }
    const std::vector<T> b = make_packed_b<T>(b_raw, nc, kc, ldb, nslots, panel);

    std::vector<T> c(static_cast<size_t>(ldc) * level + 64, from_float<T>(C_SENTINEL));
    for (int r = 0; r < level; ++r) {
        for (int col = 0; col < nc; ++col) {
            c[static_cast<size_t>(r) * ldc + col] = from_float<T>(rnd());
        }
    }
    const std::vector<T> c0 = c;

    fn(mr, nc, kc, c.data(), ldc, a.data(), b.data(), clamp_min, clamp_max);

    verify<T>(c, c0, b, b_raw, a, nc, kc, ldc, mr, level, ldb, nslots,
              clamp_min, clamp_max, zero_mode, Kind::Ab,
              [&](int k, int r) { return a[static_cast<size_t>(k) * level + r]; },
              "packed_ab");
}

/// One packed-B case: A is row-major with row pitch lda, B is packed.
template <typename T>
void check_b(BKernel<T> fn, int level, bool zero_mode, int mr, int nc, int kc,
             int lda, int ldc, float clamp_min, float clamp_max) {
    const int panel = panel_width<T>();
    const int ldb = static_cast<int>(mma_pack_align(panel * kc * sizeof(T)) / sizeof(T));
    const int nslots = panel_slots(nc, panel);

    std::vector<T> a(static_cast<size_t>(lda) * level, from_float<T>(0.f));
    for (int r = 0; r < level; ++r) {
        for (int k = 0; k < kc; ++k) {
            a[static_cast<size_t>(r) * lda + k] = from_float<T>(rnd());
        }
        for (int k = kc; k < lda; ++k) {
            a[static_cast<size_t>(r) * lda + k] = from_float<T>(77.f);
        }
    }

    std::vector<T> b_raw(static_cast<size_t>(kc) * nc);
    for (auto& v : b_raw) { v = from_float<T>(rnd()); }
    const std::vector<T> b = make_packed_b<T>(b_raw, nc, kc, ldb, nslots, panel);

    std::vector<T> c(static_cast<size_t>(ldc) * level + 64, from_float<T>(C_SENTINEL));
    for (int r = 0; r < level; ++r) {
        for (int col = 0; col < nc; ++col) {
            c[static_cast<size_t>(r) * ldc + col] = from_float<T>(rnd());
        }
    }
    const std::vector<T> c0 = c;

    fn(mr, nc, kc, c.data(), ldc, a.data(), lda, b.data(), clamp_min, clamp_max);

    verify<T>(c, c0, b, b_raw, a, nc, kc, ldc, mr, level, ldb, nslots,
              clamp_min, clamp_max, zero_mode, Kind::B,
              [&](int k, int r) { return a[static_cast<size_t>(r) * lda + k]; },
              "packed_b");
}

template <typename T>
void run_suite(Kind kind) {
    const float clamps[][2] = {{-1e30f, 1e30f}, {-0.5f, 0.5f}};
    for (int cl = 0; cl < 2; ++cl) {
        for (int nc = 0; nc <= 40; ++nc) {
            for (int kc : {1, 2, 3, 5, 7, 8, 9, 16}) {
                for (int ldc : {nc, nc + 3}) {
                    if (ldc == 0) { continue; }
                    for (int li = 0; li < MR_COUNT; ++li) {
                        const int level = MR_LEVELS[li];
                        for (int mr = 1; mr <= level; ++mr) {
                            for (bool z : {false, true}) {
                                if (kind == Kind::Ab) {
                                    check_ab<T>(ab_fn<T>(level, z), level, z, mr, nc,
                                                kc, ldc, clamps[cl][0], clamps[cl][1]);
                                } else {
                                    check_b<T>(b_fn<T>(level, z), level, z, mr, nc,
                                               kc, kc + 3, ldc,
                                               clamps[cl][0], clamps[cl][1]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

NNOPS_TEST(mma_packed_ab_f32) { run_suite<float>(Kind::Ab); }
NNOPS_TEST(mma_packed_b_f32)  { run_suite<float>(Kind::B); }
NNOPS_TEST(mma_packed_ab_f16) { run_suite<half>(Kind::Ab); }
NNOPS_TEST(mma_packed_b_f16)  { run_suite<half>(Kind::B); }

#endif  // NNOPS_ARCH_X86_64 || NNOPS_ARCH_AARCH64
